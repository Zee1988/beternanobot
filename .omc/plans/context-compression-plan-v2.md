# nanobot 上下文压缩实现计划 V3

> 基于架构评审简化方案 + 两轮深度评审修复
> 日期：2026-02-12
> 版本：V3.1（整合 v2.1 的 14 项修复 + 深度评审的 5 项修复 + 4 项优化 + 终审 5 项修复）

---

## 修改前后对比：好处与提升

### 当前问题

| 问题 | 现状 | 影响 |
|------|------|------|
| 固定滑窗 | `session/manager.py:39` 硬编码 50 条消息 | 小窗口模型浪费、大窗口模型不够用 |
| 工具结果无限增长 | `loop.py:247-249` tool result 直接追加 | 10+ 轮工具调用后上下文爆炸 |
| 异常吞噬 | `litellm_provider.py:154` 捕获所有 Exception | 溢出无法恢复，用户看到原始错误 |
| CJK 估算偏低 | `chunker.py:16` 系数 0.6 | 中文场景 token 预算不准 |
| reasoning_content 膨胀 | thinking 输出留在历史中 | DeepSeek-R1/Kimi 场景浪费大量上下文 |

### 修改后的好处

| 好处 | 说明 |
|------|------|
| **零额外 API 成本** | 所有压缩操作纯本地计算，不调用 LLM |
| **长对话不中断** | 溢出自动压缩重试，用户无感知 |
| **工具密集任务稳定** | 20+ 轮工具调用不再撑爆上下文 |
| **模型自适应** | 自动获取模型窗口大小，64K 和 1M 模型都能充分利用 |
| **向后完全兼容** | 旧 config.json 无需修改，旧 session 正常加载 |
| **维护成本极低** | 依赖 litellm 社区维护模型数据，无需自建注册表 |
| **代码量小** | 仅 ~280 行新代码，1 个新文件 + 6 个修改 |
| **全路径覆盖** | 主消息、系统消息、子代理三条 LLM 调用路径均受保护 |

### 量化提升预估

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 工具密集对话最大轮数 | ~8-12 轮（取决于结果大小） | 20 轮（max_iterations 上限） |
| 上下文溢出恢复率 | 0%（直接报错） | ~95%（自动压缩重试） |
| 64K 模型历史利用率 | 固定 50 条（可能超限） | 动态适配，扣除 system prompt 后充分利用 |
| 200K 模型历史利用率 | 固定 50 条（严重浪费） | 动态适配，扣除 system prompt 后充分利用 |
| 中文 token 估算误差 | ~30%（系数 0.6） | ~10-15%（系数 0.7 + 标点 + 降低安全系数） |

---

## 设计原则

1. **利用 litellm 已有能力**：`get_model_info()`, `token_counter()`, `ContextWindowExceededError`
2. **最少新抽象**：纯函数模块，不引入新类
3. **修复现有 bug**（异常吞噬）而非绕过它
4. **硬编码合理默认值**：只暴露 2 个配置项给用户
5. **tool_call/tool 配对安全**：所有截断操作以完整轮次为单位
6. **全路径覆盖**：`_process_message`、`_process_system_message`、`subagent` 三条路径均保护

---

## 守护条件

### 必须做到
- 所有压缩行为可通过 `context_compression: false` 关闭
- 现有 session JSONL 格式不变，旧会话可正常加载
- 现有 memory 系统不受影响
- system prompt + 最近用户消息始终完整保留
- 压缩/截断后 tool_call 与 tool result 配对完整

### 绝对不能
- 不修改 `LLMProvider` 抽象基类签名
- 不引入新的必选依赖
- 不在主消息流中引入阻塞式 LLM 调用
- 不删除或截断 system prompt

---

## Phase 1：基础设施 + 配置

### 目标
建立 token 计数、上下文窗口感知、工具结果裁剪、消息压缩的基础能力。包含"安全截断"原语，确保 tool_call/tool 配对不被破坏。

### 1.1 新建 `nanobot/agent/compressor.py`（纯函数模块，~200 行）

```python
"""Context compression utilities — pure functions, no classes."""
import copy
import json
import litellm
from loguru import logger
from nanobot.storage.chunker import estimate_tokens

# ── 常量（不暴露为配置项）──────────────────────
SOFT_TRIM_RATIO = 0.3       # 单个 tool result 占窗口 token 比例上限
HARD_TRIM_RATIO = 0.5       # 所有 tool results 占窗口 token 比例上限
SOFT_TRIM_HEAD_LINES = 20   # 软裁剪保留首 N 行
SOFT_TRIM_TAIL_LINES = 10   # 软裁剪保留尾 N 行
DEFAULT_CONTEXT_WINDOW = 128_000
REASONING_KEEP_THRESHOLD = 2000  # reasoning_content 保留阈值（字符）

def get_context_window(model: str, override: int | None = None) -> int:
    """获取模型上下文窗口大小。优先用 override，其次 litellm，最后默认值。"""
    if override is not None:  # 用 is not None 而非 truthy check，允许 override=0
        return override
    try:
        info = litellm.get_model_info(model)
        return info.get("max_input_tokens") or info.get("max_tokens") or DEFAULT_CONTEXT_WINDOW
    except Exception:
        return DEFAULT_CONTEXT_WINDOW


def count_tokens(messages: list[dict], model: str | None = None) -> int:
    """估算 messages 总 token 数。优先用 litellm，降级到启发式。"""
    if model:
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            pass
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):  # 多模态 content (list[dict])
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    total += 300  # 图片 token 估算
        if "tool_calls" in msg:
            total += estimate_tokens(json.dumps(msg["tool_calls"], ensure_ascii=False))
        rc = msg.get("reasoning_content")
        if rc and isinstance(rc, str):
            total += estimate_tokens(rc)
    # [V3] 安全系数从 1.15 降到 1.10，因为 CJK 系数已从 0.6 提到 0.7，
    # 两者叠加 1.15 * 0.7 ≈ 0.805 会导致 ~35% 过度估算
    return int(total * 1.10)

def trim_tool_result(result: str, max_chars: int = 15_000) -> str:
    """单个工具结果软裁剪：保留首尾，中间省略。"""
    if len(result) <= max_chars:
        return result
    lines = result.split("\n")
    if len(lines) <= SOFT_TRIM_HEAD_LINES + SOFT_TRIM_TAIL_LINES:
        return result[:max_chars] + f"\n...[已省略 {len(result) - max_chars} 字符]"
    head = "\n".join(lines[:SOFT_TRIM_HEAD_LINES])
    tail = "\n".join(lines[-SOFT_TRIM_TAIL_LINES:])
    omitted = len(lines) - SOFT_TRIM_HEAD_LINES - SOFT_TRIM_TAIL_LINES
    trimmed = f"{head}\n\n...[已省略 {omitted} 行]...\n\n{tail}"
    # 按行裁剪后二次检查字符数
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + f"\n...[已截断]"
    return trimmed


# ── 安全截断原语：保证 tool_call/tool 配对完整 ──

def _find_safe_cut_point(messages: list[dict], keep_last_n: int) -> int:
    """
    从后往前扫描，找到不会破坏 tool_call/tool 配对的安全切点。
    返回值是 messages 中应保留的起始索引。

    一个完整轮次 = user → assistant(可能含 tool_calls) → tool results → ...
    切点必须在一个 user 消息之前，或在一个不含 tool_calls 的 assistant 消息之前。
    """
    if keep_last_n >= len(messages):
        return 0

    # 从后往前数 keep_last_n 条消息，然后向前调整到安全点
    candidate = len(messages) - keep_last_n
    # 向前扫描，找到不会切断配对的位置
    while candidate > 0:
        msg = messages[candidate]
        role = msg.get("role", "")
        # 安全切点：user 消息之前，或不含 tool_calls 的 assistant 消息之前
        if role == "user":
            break
        if role == "assistant" and "tool_calls" not in msg:
            break
        candidate -= 1
    return max(candidate, 0)


# ── [V3] "最近一轮"定义：从最后一条 user 消息到末尾 ──

def _find_last_user_index(messages: list[dict]) -> int:
    """找到最后一条 role=user 的消息索引。找不到返回 len(messages)。"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return len(messages)


def compress_messages(
    messages: list[dict],
    context_window: int,
    model: str | None = None,
) -> list[dict]:
    """
    主压缩入口（每次 LLM 调用前执行）。
    策略：
    1. system prompt 永不裁剪
    2. 最近一轮完整对话永不裁剪（从最后一条 user 消息到末尾）[V3]
    3. reasoning_content：只保留最近一轮，其余超过阈值则移除
    4. tool results 按从旧到新：先软裁剪，再硬裁剪（达标即停）[V3]
    5. 超长 user/assistant 消息做 soft trim（>30000 字符）
    """
    # 深拷贝含嵌套结构的消息，避免污染原始 messages
    result = []
    for msg in messages:
        if "tool_calls" in msg:
            result.append(copy.deepcopy(msg))
        else:
            result.append(msg.copy())

    # [V3] 用 _find_last_user_index 精确定位"最近一轮"
    last_user_idx = _find_last_user_index(result)

    # Step 1: reasoning_content — 只保留最近一轮（last_user_idx 之后），其余超阈值则移除
    # [V3.1] 保护范围与"最近一轮"定义一致：从 last_user_idx 到末尾的所有 assistant
    for i, msg in enumerate(result):
        if i >= last_user_idx:
            continue  # 保留最近一轮所有消息的 reasoning_content
        rc = msg.get("reasoning_content")
        if rc and isinstance(rc, str) and len(rc) > REASONING_KEEP_THRESHOLD:
            del msg["reasoning_content"]

    # Step 2: 软裁剪 — 用 token 估算
    soft_token_limit = int(context_window * SOFT_TRIM_RATIO)
    for msg in result:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and estimate_tokens(content) > soft_token_limit:
                max_chars = soft_token_limit * 3  # 保守估算
                msg["content"] = trim_tool_result(content, max_chars)

    # Step 3: 硬裁剪 — 达标即停，不清除所有非最近 tool results [V3]
    hard_token_limit = int(context_window * HARD_TRIM_RATIO)
    total_tool_tokens = sum(
        estimate_tokens(msg.get("content", ""))
        for msg in result
        if msg.get("role") == "tool"
    )
    if total_tool_tokens > hard_token_limit:
        # [V3] 保护"最近一轮"：从 last_user_idx 到末尾的所有 tool results
        protected_indices = set()
        for i in range(last_user_idx, len(result)):
            if result[i].get("role") == "tool":
                protected_indices.add(i)

        # 从旧到新硬裁剪，跳过受保护的，达标即停 [V3]
        for i, msg in enumerate(result):
            if total_tool_tokens <= hard_token_limit:
                break  # [V3] 达标即停，不过度裁剪
            if msg.get("role") == "tool" and i not in protected_indices:
                old_tokens = estimate_tokens(msg.get("content", ""))
                name = msg.get("name", "tool")
                orig_len = len(msg.get("content", ""))
                placeholder = f"[工具 {name} 的结果已省略，原始长度 {orig_len} 字符]"
                msg["content"] = placeholder
                new_tokens = estimate_tokens(placeholder)
                total_tool_tokens -= (old_tokens - new_tokens)

    # Step 4: 超长 user/assistant 消息 soft trim
    # [V3] "最近一轮"保护从 last_user_idx 开始，而非 len-2
    for i, msg in enumerate(result):
        if i == 0:  # system prompt
            continue
        if i >= last_user_idx:  # [V3] 最近一轮
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 30_000:
            keep = int(30_000 * 0.6)
            tail_len = int(30_000 * 0.2)
            msg["content"] = (
                content[:keep]
                + f"\n...[已省略 {len(content) - keep - tail_len} 字符]...\n"
                + content[-tail_len:]
            )

    return result


def emergency_compress(
    messages: list[dict],
    context_window: int,
    model: str | None = None,
) -> list[dict]:
    """
    紧急压缩（溢出后调用）：
    1. 所有 tool results 替换为单行摘要
    2. 所有 reasoning_content 移除
    3. 用安全截断保留最近 5 轮完整对话
    4. 如果仍超限，安全截断到最近 2 轮
    """
    result = []
    # 保留 system prompt
    if messages and messages[0].get("role") == "system":
        result.append(messages[0].copy())
        rest = messages[1:]
    else:
        rest = messages[:]

    # 移除 reasoning_content，硬裁剪所有 tool results
    # [V3.1] 与 compress_messages 保持一致：含 tool_calls 的消息用 deepcopy
    cleaned = []
    for msg in rest:
        if "tool_calls" in msg:
            m = copy.deepcopy(msg)
        else:
            m = msg.copy()
        m.pop("reasoning_content", None)
        if m.get("role") == "tool":
            name = m.get("name", "tool")
            m["content"] = f"[{name} 结果已省略]"
        cleaned.append(m)

    # 用安全截断保留最近 ~10 条消息（不破坏配对）
    cut = _find_safe_cut_point(cleaned, 10)
    recent = cleaned[cut:]
    result.extend(recent)

    # 检查是否仍然过大
    estimated = count_tokens(result, model=model)
    if estimated > int(context_window * 0.6):
        # 极端情况：安全截断到最近 ~4 条
        result = result[:1] if result and result[0].get("role") == "system" else []
        cut2 = _find_safe_cut_point(cleaned, 4)
        result.extend(cleaned[cut2:])

    return result
```

### 1.2 修改 `nanobot/providers/base.py`（+异常类）

将 `ContextOverflowError` 定义在抽象层，而非具体实现类中，保持依赖方向正确。

位置：`base.py` 文件顶部，`LLMProvider` 类之前新增：

```python
class ContextOverflowError(Exception):
    """上下文窗口溢出，可通过压缩恢复。"""
    pass
```

### 1.3 修改 `nanobot/config/schema.py`（+2 字段）

在 `AgentDefaults` 类（`schema.py:157-163`）中新增 2 个字段：

```python
class AgentDefaults(BaseModel):
    """Default agent configuration."""
    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20
    context_compression: bool = True            # 上下文压缩总开关
    context_window_override: int | None = None  # 手动覆盖上下文窗口大小
```

### 1.4 修改 `nanobot/storage/chunker.py`（增强 CJK 估算）

修改 `estimate_tokens`（`chunker.py:7-17`），CJK 系数从 0.6 提升到 0.7，新增 CJK 标点识别：

```python
def estimate_tokens(text: str) -> int:
    """
    Simple token estimator.

    English: split by whitespace.
    CJK: each character ≈ 0.7 tokens (adjusted from 0.6).
    CJK punctuation: each ≈ 1 token.
    """
    ascii_tokens = len(re.findall(r'[a-zA-Z]+', text))
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    cjk_punct = len(re.findall(r'[\u3000-\u303f\uff00-\uffef]', text))
    other = len(re.findall(r'[0-9]+', text))
    return ascii_tokens + int(cjk_chars * 0.7) + cjk_punct + other
```

### Phase 1 验收标准
- `get_context_window("anthropic/claude-opus-4-5")` 返回 200000
- `get_context_window("unknown-model")` 返回 128000
- `get_context_window("x", override=0)` 返回 0（`is not None` 判断）
- `trim_tool_result()` 对 10000 行结果裁剪到 ~30 行，且字符数不超过 max_chars
- `_find_safe_cut_point()` 不会切断 tool_call/tool 配对
- `_find_last_user_index()` 正确定位最后一条 user 消息
- `compress_messages()` 保护 system prompt 和最近一轮消息（从最后 user 到末尾）
- `compress_messages()` 硬裁剪达标即停，不过度清除
- `emergency_compress()` 后 token 数 < 窗口的 60%，且配对完整
- `count_tokens()` 正确处理多模态 content（list[dict]），安全系数 1.10
- CJK 估算误差 < 15%
- 现有 config.json 无需修改即可加载


---

## Phase 2：集成裁剪 + 溢出恢复

### 目标
将压缩能力集成到所有 LLM 调用路径，修复 provider 异常吞噬 bug，实现溢出自动恢复。

### 2.1 修改 `nanobot/providers/litellm_provider.py`（修复异常吞噬）

修改 `chat()` 方法的 try/except 块（`litellm_provider.py:151-159`）。

关键：`litellm.ContextWindowExceededError` 必须在 `Exception` 之前捕获，否则会被吞掉。

```python
from nanobot.providers.base import ContextOverflowError  # 从 base 导入

async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
    # ... 现有代码不变（kwargs 构建、resolve_model 等）...
    try:
        response = await acompletion(**kwargs)
        return self._parse_response(response)
    except litellm.ContextWindowExceededError as e:
        # [V3] 必须在 Exception 之前！上下文溢出：向上传播，让 AgentLoop 处理
        raise ContextOverflowError(str(e)) from e
    except Exception as e:
        # 其他错误：保持现有行为
        return LLMResponse(
            content=f"Error calling LLM: {str(e)}",
            finish_reason="error",
        )
```

完整的 try/except 结构（避免歧义）：
```
try:
    response = await acompletion(**kwargs)
    return self._parse_response(response)
except litellm.ContextWindowExceededError as e:   # ← 第一个 except
    raise ContextOverflowError(str(e)) from e
except Exception as e:                             # ← 第二个 except（兜底）
    return LLMResponse(content=f"Error calling LLM: {str(e)}", finish_reason="error")
```

### 2.2 修改 `nanobot/agent/loop.py`（全路径覆盖）

#### 2.2.1 新增 import

```python
from nanobot.agent.compressor import (
    get_context_window, compress_messages, emergency_compress, trim_tool_result,
)
from nanobot.providers.base import ContextOverflowError, LLMResponse  # [V3.1] 统一导入 LLMResponse
```

#### 2.2.2 修改 `__init__`（`loop.py:38-51`）

新增 2 个参数，并初始化压缩状态：

```python
def __init__(
    self,
    bus: MessageBus,
    provider: LLMProvider,
    workspace: Path,
    model: str | None = None,
    max_iterations: int = 20,
    brave_api_key: str | None = None,
    exec_config: "ExecToolConfig | None" = None,
    cron_service: "CronService | None" = None,
    restrict_to_workspace: bool = False,
    session_manager: SessionManager | None = None,
    memory_config: dict | None = None,
    context_compression: bool = True,              # [V3] 新增
    context_window_override: int | None = None,    # [V3] 新增
    max_tokens: int = 8192,                        # [V3.1] 新增，从 config 传入
    temperature: float = 0.7,                      # [V3.1] 新增，从 config 传入
):
    # ... 现有初始化代码不变 ...
    self._compression_enabled = context_compression
    self._context_window = get_context_window(
        self.model, context_window_override
    )
    self._max_tokens = max_tokens        # [V3.1] 保存配置值，用于 LLM 调用和预算计算
    self._temperature = temperature      # [V3.1] 保存配置值
```


#### 2.2.3 新增共享 LLM 调用方法 `_call_llm_with_recovery`

[V3.1] 使用 `self._max_tokens` 和 `self._temperature` 作为默认值，与 `AgentDefaults` 配置一致：

```python
async def _call_llm_with_recovery(
    self,
    messages: list[dict],
    tools_defs: list[dict] | None = None,
    max_tokens: int | None = None,       # [V3.1] 默认用 self._max_tokens
    temperature: float | None = None,    # [V3.1] 默认用 self._temperature
) -> "LLMResponse":
    """
    共享的 LLM 调用方法，含压缩 + 溢出恢复。
    被 _process_message 和 _process_system_message 共同使用。
    """
    # [V3.1] 使用实例配置值作为默认值，而非硬编码 4096/0.7
    _max_tokens = max_tokens if max_tokens is not None else self._max_tokens
    _temperature = temperature if temperature is not None else self._temperature

    # 调用前压缩
    if self._compression_enabled:
        messages = compress_messages(messages, self._context_window, self.model)

    try:
        return await self.provider.chat(
            messages=messages,
            tools=tools_defs,
            model=self.model,
            max_tokens=_max_tokens,
            temperature=_temperature,
        )
    except ContextOverflowError:
        logger.warning("Context overflow, applying emergency compression")
        messages = emergency_compress(messages, self._context_window, self.model)
        try:
            return await self.provider.chat(
                messages=messages,
                tools=tools_defs,
                model=self.model,
                max_tokens=_max_tokens,
                temperature=_temperature,
            )
        except ContextOverflowError:
            # 二次失败：返回友好错误而非抛异常
            logger.error("Emergency compression failed, context still too large")
            return LLMResponse(
                content="对话历史过长，自动压缩后仍然超出模型限制。请发送 /clear 清空会话后重试。",
                finish_reason="error",
            )
```

#### 2.2.4 替换 `_process_message` 中的 LLM 调用（`loop.py:213-217`）

```python
# 替换原来的:
#   response = await self.provider.chat(
#       messages=messages, tools=self.tools.get_definitions(), model=self.model
#   )
# 改为:
response = await self._call_llm_with_recovery(
    messages, self.tools.get_definitions()
)
```

#### 2.2.5 替换 `_process_system_message` 中的 LLM 调用（`loop.py:328-332`）

```python
# 替换原来的:
#   response = await self.provider.chat(
#       messages=messages, tools=self.tools.get_definitions(), model=self.model
#   )
# 改为:
response = await self._call_llm_with_recovery(
    messages, self.tools.get_definitions()
)
```

#### 2.2.6 tool result 预裁剪注入点

在 `_process_message` 中（`loop.py:242` 之后，`loop.py:247` 之前）：

```python
result = await self.tools.execute(tool_call.name, tool_call.arguments)
# [V3] 预裁剪
if self._compression_enabled:
    result = trim_tool_result(result)
# ... 后续 memory.on_tool_executed 和 add_tool_result 不变
```

在 `_process_system_message` 中（`loop.py:354` 之后，`loop.py:355` 之前）：

```python
result = await self.tools.execute(tool_call.name, tool_call.arguments)  # 注意是 self.tools
# [V3] 预裁剪
if self._compression_enabled:
    result = trim_tool_result(result)
messages = self.context.add_tool_result(...)
```

### 2.3 修改 `nanobot/agent/subagent.py`（子代理保护）

子代理（`subagent.py:128`）有独立的 LLM 调用循环，也需要保护。

```python
from nanobot.agent.compressor import (
    trim_tool_result, compress_messages, emergency_compress, get_context_window,
)
from nanobot.providers.base import ContextOverflowError

# 在 _run_subagent 中（subagent.py:128 附近）：
# [V3.1] 每次 LLM 调用前执行预压缩，与主循环保护层级一致
ctx_window = get_context_window(self.model)
messages = compress_messages(messages, ctx_window, self.model)
try:
    response = await self.provider.chat(
        messages=messages, tools=tools.get_definitions(), model=self.model,
    )
except ContextOverflowError:
    logger.warning(f"Subagent [{task_id}] context overflow, emergency compress")
    messages = emergency_compress(messages, ctx_window, self.model)
    try:
        response = await self.provider.chat(
            messages=messages, tools=tools.get_definitions(), model=self.model,
        )
    except ContextOverflowError:
        # [V3] 子代理二次失败：记录错误并退出循环，而非崩溃
        logger.error(f"Subagent [{task_id}] emergency compress failed")
        final_result = "子代理上下文溢出，紧急压缩后仍超限，任务中止。"
        break

# tool result 添加时（subagent.py:157 附近）：
result = await tools.execute(tool_call.name, tool_call.arguments)
result = trim_tool_result(result)  # 预裁剪
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "name": tool_call.name,
    "content": result,
})
```


### 2.4 修改 `nanobot/cli/commands.py`（传递压缩配置）[V3 新增]

v2.1 遗漏了此文件。`AgentLoop` 有两个实例化点，都需要传递新参数。

#### Gateway 实例化（`commands.py:356-368`）

```python
agent = AgentLoop(
    bus=bus,
    provider=provider,
    workspace=config.workspace_path,
    model=config.agents.defaults.model,
    max_iterations=config.agents.defaults.max_tool_iterations,
    brave_api_key=config.tools.web.search.api_key or None,
    exec_config=config.tools.exec,
    cron_service=cron,
    restrict_to_workspace=config.tools.restrict_to_workspace,
    session_manager=session_manager,
    memory_config=config.memory.model_dump() if config.memory.enabled else None,
    context_compression=config.agents.defaults.context_compression,          # [V3]
    context_window_override=config.agents.defaults.context_window_override,  # [V3]
    max_tokens=config.agents.defaults.max_tokens,                            # [V3.1]
    temperature=config.agents.defaults.temperature,                          # [V3.1]
)
```

#### CLI agent 实例化（`commands.py:464-472`）

```python
agent_loop = AgentLoop(
    bus=bus,
    provider=provider,
    workspace=config.workspace_path,
    brave_api_key=config.tools.web.search.api_key or None,
    exec_config=config.tools.exec,
    restrict_to_workspace=config.tools.restrict_to_workspace,
    memory_config=config.memory.model_dump() if config.memory.enabled else None,
    context_compression=config.agents.defaults.context_compression,          # [V3]
    context_window_override=config.agents.defaults.context_window_override,  # [V3]
    max_tokens=config.agents.defaults.max_tokens,                            # [V3.1]
    temperature=config.agents.defaults.temperature,                          # [V3.1]
)
```

### Phase 2 验收标准
- `context_compression: true` 时，20 轮工具调用后 messages 总 token 数不超过窗口的 85%
- `context_compression: false` 时，行为与当前完全一致
- 模拟 context overflow → 自动压缩重试 → 返回正常响应
- 二次溢出失败时返回友好中文提示（非原始 API 错误）
- `_process_system_message` 路径同样受保护
- `subagent` 路径同样受保护，二次失败优雅退出而非崩溃 [V3]
- 压缩/恢复后 tool_call/tool 配对完整
- `_call_llm_with_recovery` 使用 `self._max_tokens`/`self._temperature` 作为默认值 [V3.1]
- `cli/commands.py` 两个实例化点均传递压缩配置 + max_tokens + temperature [V3.1]
- `litellm.ContextWindowExceededError` 在 `Exception` 之前被捕获 [V3]

---

## Phase 3：动态历史窗口

### 目标
将固定 50 条消息的滑窗改为基于 token 预算的动态窗口，预算考虑 system prompt 实际大小。

### 3.1 修改 `nanobot/session/manager.py`（`Session` 类）

注意：`get_history` 是 `Session` 类的方法（`manager.py:39`），不是 `SessionManager` 的方法。

[V3] 按 user+assistant 配对截断，避免截断后出现孤立的 assistant 消息：

```python
def get_history(
    self,
    max_messages: int = 50,
    max_tokens: int | None = None,  # 新增：token 预算
) -> list[dict[str, Any]]:
    """
    Get message history for LLM context.

    Args:
        max_messages: Maximum messages to return (hard cap).
        max_tokens: Optional token budget. When set, returns as many
                    recent messages as fit within this budget.
                    按 user+assistant 配对截断，不产生孤立消息。[V3]
    """
    recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
    history = [{"role": m["role"], "content": m["content"]} for m in recent]

    if max_tokens is not None and max_tokens > 0:
        from nanobot.storage.chunker import estimate_tokens
        # [V3] 按配对收集：从后往前，每次取一对 (user, assistant)
        result = []
        total = 0
        i = len(history) - 1
        while i >= 0:
            msg = history[i]
            tokens = estimate_tokens(msg.get("content", "") or "")
            # 如果是 assistant 且前一条是 user，尝试作为一对加入
            if msg.get("role") == "assistant" and i > 0 and history[i-1].get("role") == "user":
                pair_tokens = tokens + estimate_tokens(history[i-1].get("content", "") or "")
                if total + pair_tokens > max_tokens and result:
                    break
                result.append(history[i-1])  # user
                result.append(msg)           # assistant
                total += pair_tokens
                i -= 2
            else:
                if total + tokens > max_tokens and result:
                    break
                result.append(msg)
                total += tokens
                i -= 1
        return list(reversed(result))

    return history
```

注意：`get_history` 返回的是 session 中的 user/assistant 消息（不含 tool 消息），tool 消息是在 agent loop 运行时动态追加的。

### 3.2 修改 `nanobot/agent/loop.py`（历史预算）

在 `_process_message` 中（`loop.py:186-192`）和 `_process_system_message` 中（`loop.py:314-319`）：

```python
# 替换原来的: history=session.get_history()
if self._compression_enabled:
    # [V3.1] 预算使用 self._max_tokens（从配置传入），而非硬编码 8192
    history_budget = self._context_window - self._max_tokens - 4096  # 4096 safety margin
    history_budget = max(history_budget, 4096)  # 至少保留一些历史
    history = session.get_history(max_tokens=history_budget)
else:
    history = session.get_history()

messages = self.context.build_messages(
    history=history,
    current_message=msg.content,
    # ... 其余参数不变
)
```

### Phase 3 验收标准
- `get_history(max_tokens=4000)` 返回的消息总 token 数 ≤ 4000
- `get_history(max_tokens=...)` 按 user+assistant 配对截断，无孤立消息 [V3]
- `get_history()` 无参数调用行为不变（向后兼容）
- 64K 模型：历史预算 ≈ 64K - 8K - 4K = 52K tokens
- 200K 模型：历史预算 ≈ 200K - 8K - 4K = 188K tokens
- 空会话返回空列表


---

## 测试策略

### `tests/test_compressor.py`（新建）

1. **token 估算**：纯英文、纯中文、混合中英文、含 tool_calls 的 messages、多模态 content（list[dict]）、安全系数 1.10 验证
2. **上下文窗口获取**：已知模型、未知模型、override=0 边界
3. **工具结果裁剪**：短结果不裁剪、超长结果软裁剪保留首尾、按行裁剪后字符数二次检查
4. **安全截断**：`_find_safe_cut_point` 不切断 tool_call/tool 配对
5. **最近一轮定位**：`_find_last_user_index` 正确找到最后 user 消息
6. **消息压缩**：
   - system prompt 保护
   - 最近一轮保护（从最后 user 到末尾，非 len-2）[V3]
   - reasoning_content 只保留最近一轮
   - 硬裁剪达标即停（构造刚好超限的场景，验证只裁剪必要数量）[V3]
   - 硬裁剪保护最近一轮的所有 tool results
7. **紧急压缩**：压缩后 token 数在预算内、system prompt 完整、配对完整
8. **深拷贝验证**：compress_messages 不污染原始 messages
9. **空消息/单消息边界**：只有 system prompt 没有历史时的行为
10. **`context_compression: false` 回归测试**：关闭后行为与当前完全一致

### `tests/test_overflow_recovery.py`（新建）

1. **溢出检测**：`ContextOverflowError` 从 `base.py` 正确导入和抛出
2. **端到端重试**：mock LLM 第一次 overflow → 压缩 → 第二次成功
3. **二次失败友好提示**：mock LLM 两次都 overflow → 返回中文友好提示
4. **非溢出错误不触发重试**：其他 Exception 保持原有行为
5. **`_process_system_message` 路径**：系统消息处理的溢出恢复
6. **`subagent` 路径**：子代理独立循环的溢出恢复 + 二次失败优雅退出 [V3]
7. **参数透传**：`_call_llm_with_recovery` 正确传递 max_tokens/temperature [V3]
8. **异常捕获顺序**：`ContextWindowExceededError` 在 `Exception` 之前被捕获 [V3]

### `tests/test_session_history.py`（新建或追加到现有测试）

1. **配对截断**：`get_history(max_tokens=...)` 按 user+assistant 配对截断 [V3]
2. **无孤立消息**：截断后不会出现孤立的 assistant 消息 [V3]
3. **向后兼容**：无参数调用行为不变

---

## 文件变更总览

| 文件 | 操作 | Phase | V3 变更 |
|------|------|-------|---------|
| `nanobot/agent/compressor.py` | **新建** | 1 | +`_find_last_user_index`、硬裁剪达标即停、安全系数 1.10 |
| `nanobot/providers/base.py` | 修改（+异常类） | 1 | 无变化 |
| `nanobot/config/schema.py` | 修改（+2 字段） | 1 | 无变化 |
| `nanobot/storage/chunker.py` | 修改（CJK 系数） | 1 | 无变化 |
| `nanobot/providers/litellm_provider.py` | 修改（异常处理） | 2 | +完整 try/except 结构说明 |
| `nanobot/agent/loop.py` | 修改（共享方法 + 全路径） | 2, 3 | +`self._max_tokens`/`self._temperature` 从配置传入 [V3.1] |
| `nanobot/agent/subagent.py` | 修改（溢出保护） | 2 | +预压缩 + 二次失败优雅退出 [V3.1] |
| `nanobot/cli/commands.py` | 修改（传递配置） | 2 | **[V3]** 压缩参数 + **[V3.1]** max_tokens/temperature |
| `nanobot/session/manager.py` | 修改（+max_tokens） | 3 | +配对截断逻辑 |
| `tests/test_compressor.py` | **新建** | 1, 2 | +达标即停、最近一轮定位测试 |
| `tests/test_overflow_recovery.py` | **新建** | 2 | +参数透传、异常顺序、子代理二次失败测试 |

**总计**：1 个新文件 + 7 个修改 + 2 个测试文件，约 300 行新代码

---

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| token 估算不准导致仍然溢出 | 中 | 中 | Phase 2 的溢出恢复作为安全网；安全系数 1.10 |
| 工具结果裁剪丢失关键信息 | 中 | 低 | 软裁剪保留首尾；保留最近一轮所有 tool results |
| litellm API 变更 | 低 | 低 | 所有 litellm 调用有 try/except 降级 |
| 紧急压缩过于激进 | 中 | 低 | 安全截断保证配对完整；日志记录压缩详情 |
| 截断破坏 tool_call/tool 配对 | 高 | 低 | `_find_safe_cut_point` 原语 + 专项测试 |
| 紧急压缩二次失败 | 中 | 极低 | 返回友好中文提示（主循环）/ 优雅退出（子代理）[V3] |
| litellm.get_model_info 对带前缀模型名不识别 | 低 | 中 | 降级到 DEFAULT_CONTEXT_WINDOW |
| cli/commands.py 遗漏传参 | 高 | — | V3 已覆盖两个实例化点 [V3] |

---

## 实施顺序

```
Phase 1（基础设施）→ Phase 2（裁剪 + 溢出恢复）→ Phase 3（动态历史）
```

三个 Phase 有依赖关系，顺序实施。Phase 2 完成后系统已经可用（覆盖 90% 场景），Phase 3 是进一步优化。

## 未来扩展（当前不实施）

如果未来确实遇到精简方案无法覆盖的场景，可按需添加：
- 对话摘要压缩（复用现有 `SummaryTimer` 机制）
- 语义感知裁剪优先级
- 滚动摘要缓存

---

## V3 变更日志

### 继承自 v2.1 的修复（14 项）

| # | 优先级 | 修复项 |
|---|--------|--------|
| 1 | P0 | `_find_safe_cut_point()` 安全截断原语，所有截断以完整轮次为单位 |
| 2 | P0 | `_process_system_message` 通过共享 `_call_llm_with_recovery()` 获得保护 |
| 3 | P1 | `ContextOverflowError` 移到 `providers/base.py` |
| 4 | P1 | 二次溢出失败返回友好中文提示 |
| 5 | P1 | `compress_messages` 对含 `tool_calls` 的消息使用 `copy.deepcopy` |
| 6 | P1 | 裁剪阈值改用 `estimate_tokens()` 而非 `token * 4` 字符数 |
| 7 | P2 | `emergency_compress` 增加 `model` 参数 |
| 8 | P2 | 历史预算改为 `context_window - max_tokens - safety_margin` |
| 9 | P2 | `reasoning_content` 改为"只保留最近一轮"，阈值 2000 |
| 10 | P2 | `count_tokens` 处理 `list` 类型 content（多模态） |
| 11 | P2 | `subagent.py` 添加溢出保护和 tool result 预裁剪 |
| 12 | P3 | `get_context_window` 的 `override` 判断改为 `is not None` |
| 13 | P3 | `trim_tool_result` 按行裁剪后做字符数二次检查 |
| 14 | P3 | 硬裁剪改为"保留最近一轮 assistant 的所有 tool results" |

### V3 新增修复（5 项）

| # | 优先级 | 修复项 | 来源 |
|---|--------|--------|------|
| 15 | P0 | `_call_llm_with_recovery` 接受并透传 `max_tokens`/`temperature` | 深度评审 P0-1 |
| 16 | P0 | 硬裁剪达标即停，不清除所有非最近 tool results | 深度评审 P0-2 |
| 17 | P1 | 明确 `get_history()` 是 `Session` 类方法，非 `SessionManager` | 深度评审 P1-1 |
| 18 | P1 | `cli/commands.py` 两个 `AgentLoop` 实例化点传递压缩配置 | 深度评审 P1-2 |
| 19 | P1 | 明确 `litellm.ContextWindowExceededError` 必须在 `Exception` 之前捕获 | 深度评审 P1-3 |

### V3 新增优化（4 项）

| # | 优化项 | 来源 |
|---|--------|------|
| 20 | `count_tokens` 安全系数从 1.15 降到 1.10，避免与 CJK 0.7 叠加过度估算 | 深度评审 OPT-1 |
| 21 | "最近一轮"保护改为从最后 user 消息到末尾（`_find_last_user_index`），替代 `len-2` | 深度评审 OPT-2 |
| 22 | 子代理二次溢出失败优雅退出（设置 final_result + break），而非崩溃 | 深度评审 OPT-3 |
| 23 | `get_history(max_tokens=...)` 按 user+assistant 配对截断，避免孤立消息 | 深度评审 OPT-4 |

### V3.1 终审修复（5 项）

| # | 优先级 | 修复项 | 来源 |
|---|--------|--------|------|
| 24 | P0 | `_call_llm_with_recovery` 默认值改用 `self._max_tokens`/`self._temperature`（从配置传入），而非硬编码 4096/0.7 | 终审 P0-1 |
| 25 | P0 | `emergency_compress` 对含 `tool_calls` 的消息使用 `copy.deepcopy`，与 `compress_messages` 一致 | 终审 P0-2 |
| 26 | P0 | `_process_system_message` 预裁剪代码修正为 `self.tools.execute`（非 `tools`） | 终审 P0-3 |
| 27 | P1 | `reasoning_content` 保护范围改为 `i >= last_user_idx`，与"最近一轮"定义一致 | 终审 P1-1 |
| 28 | P1 | `LLMResponse` 移到 `loop.py` 顶部统一导入，移除函数体内延迟导入 | 终审 P1-3 |

### V3.1 终审优化（2 项）

| # | 优化项 | 来源 |
|---|--------|------|
| 29 | 历史预算 `max_output` 改用 `self._max_tokens`（从配置传入），而非硬编码 8192 | 终审 P1-4 |
| 30 | subagent 每次 LLM 调用前增加 `compress_messages` 预压缩，与主循环保护层级一致 | 终审 P2-4 |
