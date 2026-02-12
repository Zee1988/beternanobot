# 上下文压缩方案架构评审

> 评审角色：Strategic Architecture Advisor
> 评审日期：2026-02-12
> 评审对象：`context-compression-plan.md`（9 阶段方案）
> 评审维度：成本、可行性、复杂度、健壮性

---

## 一、总体判断

当前 9 阶段方案**严重过度工程化**。nanobot 是一个轻量级个人 AI 助手框架，核心代码量约 2000 行，当前只有 2 个测试文件。该方案引入了 6 个新文件、3 个新抽象层（`ContextCompressor`、`MessageClassifier`、`RollingSummary`）、修改 8 个现有文件，复杂度远超问题本身。

实际问题可以归结为三件事：
1. 工具结果太大时会撑爆上下文（最常见）
2. 历史消息无 token 预算控制（次常见）
3. 溢出后直接报错，没有恢复机制（安全网）

这三个问题用 **3 个阶段、2 个新文件** 就能解决。阶段六到九（语义分类、非工具裁剪、滚动摘要、结构化摘要）在当前项目规模下属于 YAGNI（You Aren't Gonna Need It）。

---

## 二、逐阶段评审

### 阶段一：Token 计数与上下文窗口感知 — 部分合理，但有冗余

**问题 1：自建 `model_limits.py` 是重复造轮子**

litellm 已经内置了 `litellm.get_max_tokens(model)` 和 `litellm.model_cost` 字典，覆盖了计划中列出的所有模型。nanobot 已经依赖 litellm（见 `litellm_provider.py:8`），没有理由自己维护一份模型窗口注册表。

自建注册表的维护成本：每当新模型发布（Claude 4、GPT-5 等），需要手动更新字典。litellm 的注册表由社区维护，更新更及时。

```python
# 已有的 litellm 能力，无需自建
import litellm
litellm.get_max_tokens("anthropic/claude-opus-4-5")  # -> 200000
litellm.get_max_tokens("deepseek/deepseek-chat")     # -> 64000
```

**问题 2：tiktoken 可选依赖增加复杂度**

计划要求"当 tiktoken 可用时自动切换为精确计数"。这引入了两条代码路径（精确 vs 启发式），增加测试负担。对于上下文压缩的决策点，启发式估算乘以 1.2 安全系数已经足够——溢出恢复（阶段三）是真正的安全网。

**问题 3：`count_message_tokens()` 的设计合理**

增强 `chunker.py` 的 token 估算，支持完整 messages 列表（含 tool_calls JSON），这是必要的基础设施。

**结论**：保留 token 估算增强，删除 `model_limits.py`，改用 litellm 内置能力。

### 阶段二：工具结果裁剪 — 核心价值，但设计过重

**合理之处**：
- 工具结果裁剪是 ROI 最高的优化，这个判断正确
- 两阶段裁剪（软/硬）策略合理
- 文件操作保留路径的细节考虑周到

**问题 1：`ContextCompressor` 类承担了太多职责**

计划让 `ContextCompressor` 同时负责：工具结果裁剪、消息压缩、token 估算、紧急压缩、历史摘要。这违反了单一职责原则，且后续阶段不断往这个类里塞新方法。

更好的做法：工具结果裁剪是一组纯函数，不需要类。只需要一个 `trim_tool_result()` 和一个 `compress_messages()` 函数，加上配置参数。

**问题 2：配置项过多**

计划在 `AgentDefaults` 中新增 6 个字段（`context_compression`、`context_window_override`、`tool_result_soft_limit`、`tool_result_hard_limit`、`context_reserve_ratio`、`compression_model`），后续阶段还要加 `enable_history_summarization`、`max_user_message_chars`、`max_assistant_message_chars`。

对于个人助手框架，用户不需要调 9 个旋钮。合理的配置是：一个总开关 + 一个窗口覆盖值，其余用硬编码的合理默认值。

**结论**：保留核心裁剪逻辑，简化为函数模块而非类，减少配置项。

### 阶段三：溢出恢复与自动重试 — 必要，但实现方式有侵入性

**问题 1：修改 `LLMProvider` 抽象接口是高风险操作**

计划要在 `LLMProvider.chat()` 的签名中添加 `on_overflow` 回调参数。这修改了抽象基类（`providers/base.py:44-65`），影响所有 provider 实现。对于一个只有一个实现（`LiteLLMProvider`）的接口来说，这种"面向未来"的设计是过度的。

更好的做法：溢出检测和重试逻辑放在 `AgentLoop` 层，不修改 provider 接口。agent loop 已经有 try/except 处理错误响应的逻辑（`loop.py:154`），在这里捕获溢出错误、压缩、重试，是最自然的位置。

```python
# 在 AgentLoop 中处理，不污染 provider 接口
try:
    response = await self.provider.chat(messages=messages, ...)
except ContextOverflowError:
    messages = emergency_compress(messages)
    response = await self.provider.chat(messages=messages, ...)
```

**问题 2：当前 `LiteLLMProvider.chat()` 吞掉了所有异常**

`litellm_provider.py:151-159` 的 except 块捕获所有 Exception 并返回错误字符串，这意味着 `AgentLoop` 永远看不到异常。这是需要修复的真正 bug——应该让 context overflow 异常向上传播，而不是在 provider 层加回调。

**结论**：保留溢出恢复，但改为在 AgentLoop 层实现；修复 provider 的异常吞噬问题。

### 阶段四：动态历史窗口 — 合理且低成本

这是一个简单且有效的改进。修改 `Session.get_history()` 支持 token 预算，改动小、风险低、向后兼容。

**唯一问题**：计划中的 token 预算计算公式过于复杂：
```python
history_budget = int(window * (1 - reserve)) - system_tokens - 2000
```
这个 2000 的魔法数字（"for current msg + memory"）不够健壮。更好的做法是直接用窗口的固定比例（如 60%）作为历史预算，简单可靠。

**结论**：保留，简化预算计算。

### 阶段五：对话摘要压缩 — 过早优化，建议删除

**成本问题**：每次摘要需要一次 LLM API 调用。对于个人助手，大多数对话不会长到需要摘要。计划自己也标注了"低优先级"和"默认关闭"——如果默认关闭，说明当前不需要。

**与现有 memory 系统冲突**：nanobot 已经有 `MemoryStore`（`agent/memory.py`）和 `SummaryTimer`（`storage/summary.py`），负责对话摘要和记忆索引。再加一层摘要压缩，职责边界模糊。

**结论**：删除。如果未来确实需要，可以复用现有的 `SummaryTimer` 机制。

### 阶段六：语义感知裁剪优先级 — YAGNI

**问题**：引入 `MessageClassifier` 和 `MessagePriority` 枚举，用关键词规则对消息分类。这是一个看起来优雅但实际收益极低的抽象。

实际场景中，工具结果裁剪按时间顺序（旧的先裁）已经足够好。用户说"请记住以后都用 TypeScript"这种指令，会被 memory 系统（`MemoryStore`）持久化，不依赖上下文窗口中的历史消息存活。

引入分类器的代价：新文件、新枚举、compressor 排序逻辑变复杂、需要维护关键词列表（中英文双语）、边界 case 多（"重要"出现在 tool result 的文件内容中怎么办？）。

**结论**：删除。时间顺序裁剪 + memory 系统已经覆盖了这个需求。

### 阶段七：扩展 Pruning 到非 Tool Result 消息 — 有价值但可简化

用户粘贴大段代码确实会撑爆上下文，这个问题真实存在。但不需要单独一个阶段——可以合并到阶段二的裁剪逻辑中，作为 `compress_messages()` 的一部分。

**结论**：合并到阶段二，不单独成阶段。

### 阶段八：分层摘要与摘要缓存 — 过度工程化

引入 `RollingSummary` 类，每 10 轮对话生成增量摘要，缓存到 session metadata。这是一个复杂的有状态系统，需要处理：
- 摘要触发时机
- 增量 vs 全量摘要合并
- session metadata 格式扩展
- LLM 调用失败的降级
- 与现有 `SummaryTimer` 的职责划分

对于个人助手，大多数对话在 10 轮以内就结束了。需要 20+ 轮的场景（工具密集型任务）已经被阶段二的工具裁剪覆盖。

**结论**：删除。

### 阶段九：结构化摘要格式 — 过度工程化

依赖阶段八的 `RollingSummary`，阶段八已删除，此阶段自然取消。即使保留，让 LLM 输出结构化 JSON 摘要并可靠解析，本身就是一个脆弱的设计——LLM 输出格式不稳定，JSON 解析失败的降级路径增加复杂度。

**结论**：删除。

---

## 三、被忽视的关键问题

### 3.1 Provider 异常吞噬（当前最严重的 bug）

`litellm_provider.py:151-159` 的 except 块：

```python
except Exception as e:
    return LLMResponse(
        content=f"Error calling LLM: {str(e)}",
        finish_reason="error",
    )
```

这段代码把所有异常（包括 context overflow）转换为正常响应返回。`AgentLoop` 检查 `response.has_tool_calls` 和 `response.content`，会把错误信息当作正常回复发给用户。

这意味着：
1. 溢出恢复无法在 AgentLoop 层通过 try/except 实现（异常被吞了）
2. 用户会收到 "Error calling LLM: ..." 这样的原始错误信息
3. 任何 API 错误（限流、认证失败、网络超时）都被静默处理

**修复方案**：区分可恢复错误和不可恢复错误。context overflow 应该抛出特定异常让上层处理；其他错误可以保留当前行为或也向上传播。

### 3.2 litellm 已有的能力未被利用

litellm 提供了以下能力，计划完全没有提及：

| litellm 能力 | 计划中的替代方案 | 建议 |
|---|---|---|
| `litellm.get_max_tokens(model)` | 自建 `model_limits.py` | 用 litellm |
| `litellm.token_counter(model, messages)` | 自建 `count_message_tokens()` | 用 litellm，降级到自建估算 |
| `litellm.ContextWindowExceededError` | 字符串模式匹配 | 直接捕获此异常类 |

### 3.3 `reasoning_content` 的上下文膨胀

`context.py:231-233` 将 `reasoning_content` 存入 messages 历史。DeepSeek-R1 和 Kimi 的 thinking 输出可能非常长（数千 token），但计划中只在"紧急压缩"时才移除。应该在常规压缩中就处理。

---

## 四、替代方案：3 阶段精简版

### 设计原则

1. **利用 litellm 已有能力**，不重复造轮子
2. **最少新抽象**：函数优于类，硬编码合理默认值优于配置项
3. **修复现有 bug**（异常吞噬）而非绕过它
4. **阶段二的裁剪是核心**，其余都是辅助

### Phase 1：基础设施 + 配置（1 个新文件，2 个修改）

**新建 `nanobot/agent/compressor.py`**（纯函数模块，约 120 行）：

```python
"""Context compression utilities."""
import litellm
from nanobot.storage.chunker import estimate_tokens

def get_context_window(model: str) -> int:
    """获取模型上下文窗口大小，优先用 litellm，降级到默认值。"""
    try:
        info = litellm.get_model_info(model)
        return info.get("max_input_tokens") or info.get("max_tokens") or 128_000
    except Exception:
        return 128_000

def count_tokens(messages: list[dict], model: str | None = None) -> int:
    """估算 messages 总 token 数。优先用 litellm，降级到启发式。"""
    if model:
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            pass
    # 降级：启发式估算
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += estimate_tokens(content)
        # tool_calls JSON
        if "tool_calls" in msg:
            import json
            total += estimate_tokens(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return int(total * 1.15)  # 安全系数

def trim_tool_result(result: str, max_chars: int = 15000) -> str:
    """单个工具结果软裁剪：保留首尾，中间省略。"""
    if len(result) <= max_chars:
        return result
    lines = result.split("\n")
    if len(lines) <= 40:
        return result[:max_chars] + f"\n...[已省略 {len(result) - max_chars} 字符]"
    head = "\n".join(lines[:20])
    tail = "\n".join(lines[-10:])
    omitted = len(lines) - 30
    return f"{head}\n\n...[已省略 {omitted} 行]...\n\n{tail}"

def compress_messages(
    messages: list[dict],
    context_window: int,
    model: str | None = None,
) -> list[dict]:
    """
    主压缩入口。策略：
    1. system prompt 永不裁剪
    2. 最近一轮 user+assistant 永不裁剪
    3. reasoning_content 超过 500 字符则移除
    4. tool results 按从旧到新裁剪
    5. 超长 user/assistant 消息做 soft trim
    """
    ...

def emergency_compress(messages: list[dict], context_window: int) -> list[dict]:
    """紧急压缩：所有 tool results 替换为单行摘要，只保留最近 3 轮。"""
    ...
```

**修改 `nanobot/config/schema.py`**（+2 个字段）：

```python
class AgentDefaults(BaseModel):
    # ... 现有字段 ...
    context_compression: bool = True
    context_window_override: int | None = None
```

只需要 2 个配置项。裁剪阈值（soft 30%、hard 50%、reserve 15%）作为 `compressor.py` 内的常量，不暴露给用户。

**修改 `nanobot/storage/chunker.py`**（增强 `estimate_tokens`）：

- CJK 系数从 0.6 调整为 0.7
- 增加标点和 emoji 的基础处理

### Phase 2：集成裁剪 + 溢出恢复（2 个修改）

**修改 `nanobot/providers/litellm_provider.py`**（修复异常吞噬）：

```python
# 新增异常类
class ContextOverflowError(Exception):
    """上下文窗口溢出，可通过压缩恢复。"""
    pass

async def chat(self, messages, ...):
    try:
        response = await acompletion(**kwargs)
        return self._parse_response(response)
    except litellm.ContextWindowExceededError as e:
        raise ContextOverflowError(str(e)) from e
    except Exception as e:
        return LLMResponse(content=f"Error calling LLM: {str(e)}", finish_reason="error")
```

注意：只让 context overflow 向上传播，其他错误保持现有行为，最小化变更风险。不修改 `LLMProvider` 抽象基类的签名。

**修改 `nanobot/agent/loop.py`**（集成压缩 + 溢出恢复）：

```python
from nanobot.agent.compressor import (
    get_context_window, compress_messages, emergency_compress,
    trim_tool_result, count_tokens,
)
from nanobot.providers.litellm_provider import ContextOverflowError

# __init__ 中：
self._context_window = config_override or get_context_window(self.model)
self._compression_enabled = agent_defaults.context_compression

# agent loop 中，每次 LLM 调用前：
if self._compression_enabled:
    messages = compress_messages(messages, self._context_window, self.model)

# agent loop 中，LLM 调用时：
try:
    response = await self.provider.chat(messages=messages, ...)
except ContextOverflowError:
    logger.warning("Context overflow, applying emergency compression")
    messages = emergency_compress(messages, self._context_window)
    response = await self.provider.chat(messages=messages, ...)

# tool result 添加时：
if self._compression_enabled:
    result = trim_tool_result(result)
```

### Phase 3：动态历史窗口（2 个修改）

**修改 `nanobot/session/manager.py`**：

给 `Session.get_history()` 加 `max_tokens` 参数（向后兼容）。

**修改 `nanobot/agent/loop.py`**：

```python
if self._compression_enabled:
    history_budget = int(self._context_window * 0.6)
    history = session.get_history(max_tokens=history_budget)
else:
    history = session.get_history()
```

### 测试（2 个新文件）

- `tests/test_compressor.py`：token 估算、工具裁剪、消息压缩、紧急压缩
- `tests/test_overflow_recovery.py`：溢出检测、重试流程

---

## 五、方案对比

| 维度 | 原 9 阶段方案 | 精简 3 阶段方案 |
|------|-------------|---------------|
| **新文件数** | 6 | 1（+ 2 测试） |
| **修改文件数** | 8 | 4 |
| **新抽象/类** | 3（ContextCompressor, MessageClassifier, RollingSummary） | 0（纯函数） |
| **新配置项** | 9 | 2 |
| **额外 LLM API 调用** | 是（摘要压缩、滚动摘要） | 否 |
| **预估代码量** | ~800 行 | ~200 行 |
| **预估开发时间** | 3-5 天 | 1-1.5 天 |
| **维护负担** | 高（模型注册表、关键词列表、摘要缓存） | 低（litellm 维护模型数据） |
| **核心问题覆盖** | 全部 3 个 | 全部 3 个 |
| **向后兼容** | 是 | 是 |
| **CJK 处理** | 是 | 是（复用增强后的 chunker） |

### 成本分析

| 成本类型 | 原方案 | 精简方案 |
|---------|-------|---------|
| 开发成本 | 高（9 阶段，多文件协调） | 低（3 阶段，改动集中） |
| API 成本 | 有增量（摘要调用） | 零增量 |
| 维护成本 | 高（自建注册表、分类器规则） | 低（依赖 litellm 社区维护） |
| 调试成本 | 高（多层压缩、优先级排序） | 低（线性逻辑，易追踪） |

### 风险对比

| 风险 | 原方案 | 精简方案 |
|------|-------|---------|
| 裁剪丢失关键信息 | 中（语义分类可能误判） | 低（简单时间顺序，可预测） |
| 摘要质量不稳定 | 高（依赖 LLM 输出格式） | 无（不做摘要） |
| litellm 版本兼容 | 中（自建注册表可能过时） | 低（直接用 litellm API） |
| 回归风险 | 高（改 8 个文件） | 中（改 4 个文件） |

---

## 六、最终建议

**采用精简 3 阶段方案**，理由：

1. **解决了全部 3 个核心问题**（工具裁剪、历史预算、溢出恢复），覆盖了 95% 的实际场景
2. **代码量是原方案的 1/4**，开发和维护成本显著降低
3. **不引入额外 API 成本**，对个人用户友好
4. **利用 litellm 已有能力**，减少自维护的代码
5. **修复了原方案忽视的 bug**（provider 异常吞噬）

如果未来确实遇到精简方案无法覆盖的场景（如超长对话需要摘要），可以在 Phase 3 之后按需添加。但根据 nanobot 的定位（轻量级个人助手），这种场景的概率很低。

### 实施优先级

```
Phase 1（基础设施）→ Phase 2（裁剪 + 溢出恢复）→ Phase 3（动态历史）
     0.5 天                  0.5 天                    0.5 天
```

三个 Phase 之间有依赖关系，建议顺序实施。Phase 2 完成后系统已经可用，Phase 3 是锦上添花。

---

## 七、参考文件

| 文件 | 关键发现 |
|------|---------|
| `nanobot/agent/loop.py:209-256` | agent loop 主循环，工具结果直接追加无裁剪 |
| `nanobot/agent/loop.py:186-192` | `build_messages()` 调用点，历史无 token 预算 |
| `nanobot/providers/litellm_provider.py:151-159` | 异常吞噬 bug，所有错误转为字符串响应 |
| `nanobot/providers/base.py:44-65` | `LLMProvider.chat()` 抽象接口，不应修改签名 |
| `nanobot/session/manager.py:39-53` | `get_history()` 硬编码 50 条，无 token 感知 |
| `nanobot/storage/chunker.py:7-17` | `estimate_tokens()` CJK 系数 0.6 偏低 |
| `nanobot/agent/context.py:179-204` | `add_tool_result()` 无大小限制 |
| `nanobot/agent/memory.py:149-169` | `recall()` 已有 150ms 超时和 800 token 预算 |
| `nanobot/config/schema.py:157-164` | `AgentDefaults` 当前 5 个字段，不宜膨胀 |
| `nanobot/providers/registry.py` | 12 个 provider，litellm 已覆盖其模型信息 |
