# nanobot 上下文压缩实现计划

## 背景

当前 nanobot 的上下文管理存在以下问题：

1. **固定滑动窗口**：`Session.get_history()` 硬编码 50 条消息，不感知模型上下文窗口大小
2. **无 token 预算管理**：发送给 LLM 的 messages 列表没有 token 总量控制
3. **无溢出恢复**：`LiteLLMProvider.chat()` 遇到 context overflow 错误直接返回错误字符串，不尝试压缩重试
4. **工具结果无限增长**：agent loop 中 tool result 直接追加到 messages，长对话中工具结果可能占据大量上下文
5. **token 估算粗糙**：`chunker.py` 的 `estimate_tokens` 仅用于 memory 注入，未用于主对话流

参考实现 (openclaw) 的核心策略：
- 两阶段工具结果裁剪（软裁剪保留首尾 + 硬裁剪清空）
- 基于上下文窗口比例的动态阈值（30%/50%）
- 溢出后自动压缩 -> 裁剪 -> 重试链
- 模型感知的窗口大小解析

## 工作目标

为 nanobot 添加渐进式上下文压缩能力，确保：
- 长对话不会因 context overflow 而中断
- 工具密集型任务不会浪费上下文窗口
- 对现有会话和配置完全向后兼容
- CJK（中文）场景下 token 估算准确

---

## 守护条件

### 必须做到
- 所有压缩行为可通过 `config.json` 开关控制
- 现有 session JSONL 格式不变，旧会话可正常加载
- 现有 memory 系统不受影响
- 压缩后保留 system prompt + 最近用户消息的完整性
- 支持所有已注册的 LLM provider

### 绝对不能
- 不能修改 `Session` 的持久化格式
- 不能引入新的必选依赖
- 不能在主消息流中引入阻塞式 LLM 调用（压缩摘要除外）
- 不能删除或截断 system prompt

---

## 阶段一：Token 计数与上下文窗口感知

### 目标
建立准确的 token 计数基础设施，让系统知道"当前上下文有多大"和"模型能容纳多大"。

### 修改文件

#### 1.1 增强 token 估算器 — `nanobot/storage/chunker.py`

**当前状态**：`estimate_tokens()` 使用简单正则，CJK 按 0.6 系数估算，无标点/emoji 处理。

**改动**：
- 新增 `count_message_tokens(messages: list[dict]) -> int` 函数，计算完整 messages 列表的 token 数
- 改进 CJK 估算：中文字符系数从 0.6 调整为 0.7（更接近 tiktoken 实测值）
- 增加对 tool_calls JSON 结构的 token 估算（function name + arguments 序列化）
- 增加对 `reasoning_content` 字段的估算（thinking 模型）
- 当 `tiktoken` 可用时自动切换为精确计数（关键决策点如 compaction 阈值判断、overflow 预测必须用精确值；日常 pruning 可用启发式）

**验收标准**：
- `count_message_tokens()` 对包含 system/user/assistant/tool 四种角色的 messages 返回合理估算
- 中文文本估算误差 < 20%（与 tiktoken cl100k_base 对比）
- 无 tiktoken 时优雅降级到启发式估算

#### 1.2 模型上下文窗口注册表 — `nanobot/providers/model_limits.py`（新建）

**改动**：
- 创建 `MODEL_CONTEXT_WINDOWS` 字典，映射模型名称模式到上下文窗口大小
- 覆盖主要模型：Claude (200K), GPT-4o (128K), DeepSeek (64K/128K), Gemini (1M/2M), Qwen (128K), Kimi (128K), MiniMax (1M), GLM-4 (128K)
- 提供 `get_context_window(model: str) -> int` 函数，支持模糊匹配
- 默认值：128000（安全保守值）
- 提供 `get_output_limit(model: str) -> int` 函数

**验收标准**：
- `get_context_window("anthropic/claude-opus-4-5")` 返回 200000
- `get_context_window("deepseek-chat")` 返回 64000
- 未知模型返回 128000

#### 1.3 配置扩展 — `nanobot/config/schema.py`

**改动**：在 `AgentDefaults` 中新增字段：

```python
class AgentDefaults(BaseModel):
    # ... 现有字段 ...
    context_compression: bool = True           # 总开关
    context_window_override: int | None = None # 手动覆盖上下文窗口大小
    tool_result_soft_limit: float = 0.3        # 单个工具结果占窗口比例上限（软裁剪）
    tool_result_hard_limit: float = 0.5        # 所有工具结果占窗口比例上限（硬裁剪）
    context_reserve_ratio: float = 0.15        # 为输出预留的窗口比例
    compression_model: str | None = None       # 用于摘要压缩的模型（None=使用当前模型）
```

**验收标准**：
- 现有 config.json 无需修改即可正常加载（所有新字段有默认值）
- `context_compression: false` 时系统行为与当前完全一致

---

## 阶段二：工具结果裁剪（实时、无 LLM 调用）

### 目标
在 agent loop 的每次迭代中，对工具结果进行智能裁剪，防止单个大结果或累积结果撑爆上下文。这是最高 ROI 的优化——不需要额外 API 调用，立即生效。

### 新建文件

#### 2.1 上下文压缩器 — `nanobot/agent/compressor.py`（新建）

**核心类**：`ContextCompressor`

```python
class ContextCompressor:
    def __init__(
        self,
        context_window: int,
        soft_ratio: float = 0.3,    # 单个 tool result 上限
        hard_ratio: float = 0.5,    # 全部 tool results 上限
        reserve_ratio: float = 0.15, # 输出预留
    ): ...

    def trim_tool_result(self, result: str, max_chars: int | None = None) -> str:
        """单个工具结果裁剪：保留首尾，中间用摘要替代"""

    def compress_messages(self, messages: list[dict]) -> list[dict]:
        """
        主入口：对 messages 列表进行就地压缩。
        策略：
        1. system prompt 永不裁剪
        2. 最近一轮 user 消息永不裁剪
        3. 工具结果按新旧排序，旧的优先裁剪
        4. 两阶段：先软裁剪（保留首尾各 N 行），再硬裁剪（替换为 "[结果已省略]"）
        """

    def estimate_total_tokens(self, messages: list[dict]) -> int:
        """估算 messages 总 token 数"""
```

**裁剪策略细节**：

- **软裁剪**（Phase 1）：当单个 tool result 超过 `context_window * soft_ratio` 的字符数时，保留首 20 行 + 尾 10 行，中间替换为 `\n...[已省略 {N} 行，约 {T} tokens]...\n`
- **硬裁剪**（Phase 2）：当所有 tool results 总量超过 `context_window * hard_ratio` 时，从最旧的 tool result 开始，替换为 `[工具 {name} 的结果已省略，原始长度约 {T} tokens]`
- **文件操作保护**：`read_file` / `write_file` / `edit_file` 的结果在硬裁剪时保留路径信息

**验收标准**：
- 单个 10000 行的 tool result 被软裁剪到 ~30 行
- 累积 tool results 超过窗口 50% 时触发硬裁剪
- system prompt 和最近 user 消息始终完整保留
- 文件操作结果裁剪后仍保留文件路径

### 修改文件

#### 2.2 集成到 agent loop — `nanobot/agent/loop.py`

**改动**：

1. 在 `AgentLoop.__init__()` 中初始化 `ContextCompressor`：
```python
from nanobot.agent.compressor import ContextCompressor
from nanobot.providers.model_limits import get_context_window

# 在 __init__ 中
window = config_override or get_context_window(self.model)
self.compressor = ContextCompressor(
    context_window=window,
    soft_ratio=agent_defaults.tool_result_soft_limit,
    hard_ratio=agent_defaults.tool_result_hard_limit,
    reserve_ratio=agent_defaults.context_reserve_ratio,
) if agent_defaults.context_compression else None
```

2. 在 agent loop 的每次 LLM 调用前插入压缩：
```python
# 在 while iteration < self.max_iterations 循环内，调用 provider.chat() 之前
if self.compressor:
    messages = self.compressor.compress_messages(messages)
```

3. 在 `context.add_tool_result()` 时对单个结果预裁剪：
```python
if self.compressor:
    result = self.compressor.trim_tool_result(result)
```

**验收标准**：
- `context_compression: true` 时，20 轮工具调用后 messages 总 token 数不超过窗口的 85%
- `context_compression: false` 时，行为与当前完全一致
- 不影响 `_process_system_message` 的子代理消息处理

---

## 阶段三：溢出恢复与自动重试

### 目标
当 LLM API 返回 context overflow 错误时，自动执行压缩并重试，而不是直接返回错误给用户。

### 修改文件

#### 3.1 错误检测与重试 — `nanobot/providers/litellm_provider.py`

**当前状态**：`chat()` 方法的 except 块直接返回 `LLMResponse(content=f"Error calling LLM: {str(e)}", finish_reason="error")`，不区分错误类型。

**改动**：

1. 新增 `_is_context_overflow(error: Exception) -> bool` 方法：
   - 检测 litellm 的 `ContextWindowExceededError`
   - 检测常见错误消息模式：`"context_length_exceeded"`, `"maximum context length"`, `"too many tokens"`, `"prompt is too long"`
   - 覆盖各 provider 的不同错误格式

2. 修改 `chat()` 方法，增加 `on_overflow` 回调参数：
```python
async def chat(
    self,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    on_overflow: Callable[[list[dict]], list[dict]] | None = None,  # 新增
) -> LLMResponse:
```

3. 在 except 块中，如果检测到 context overflow 且 `on_overflow` 不为 None：
   - 调用 `on_overflow(messages)` 获取压缩后的 messages
   - 用压缩后的 messages 重试一次
   - 如果重试仍然失败，返回错误

**注意**：`on_overflow` 回调由 `AgentLoop` 提供，内部调用 `ContextCompressor` 的紧急压缩模式。

**验收标准**：
- 模拟 context overflow 错误时，自动触发压缩重试
- 重试成功时用户无感知
- 重试失败时返回清晰的错误信息（而非原始 API 错误）
- `on_overflow` 为 None 时行为与当前一致（向后兼容）

#### 3.2 紧急压缩模式 — `nanobot/agent/compressor.py`

**改动**：新增 `emergency_compress()` 方法：

```python
def emergency_compress(self, messages: list[dict]) -> list[dict]:
    """
    溢出后的紧急压缩，比常规压缩更激进：
    1. 所有 tool results 硬裁剪为单行摘要
    2. 历史对话只保留最近 5 轮
    3. reasoning_content 全部移除
    4. 如果仍然超限，移除中间历史只保留 system + 最近 2 轮
    """
```

**验收标准**：
- 紧急压缩后 messages 总 token 数 < 窗口的 60%
- system prompt 完整保留
- 最近一轮 user 消息完整保留

#### 3.3 集成到 agent loop — `nanobot/agent/loop.py`

**改动**：在 `_process_message` 和 `_process_system_message` 中，传递 overflow 回调：

```python
response = await self.provider.chat(
    messages=messages,
    tools=self.tools.get_definitions(),
    model=self.model,
    on_overflow=self.compressor.emergency_compress if self.compressor else None,
)
```

**验收标准**：
- 端到端测试：构造超长 messages -> 触发 overflow -> 自动压缩重试 -> 返回正常响应

---

## 阶段四：动态历史窗口

### 目标
将 `Session.get_history()` 的固定 50 条消息改为基于 token 预算的动态窗口，让历史消息量自适应模型容量和当前上下文占用。

### 修改文件

#### 4.1 动态历史裁剪 — `nanobot/session/manager.py`

**改动**：

1. 修改 `Session.get_history()` 签名：
```python
def get_history(
    self,
    max_messages: int = 50,
    max_tokens: int | None = None,  # 新增：token 预算
) -> list[dict[str, Any]]:
```

2. 当 `max_tokens` 不为 None 时：
   - 从最新消息向前遍历
   - 累加每条消息的 token 估算
   - 当累计超过 `max_tokens` 时停止
   - 返回在预算内的最近 N 条消息

3. 保持 `max_messages` 作为硬上限（即使 token 预算未满也不超过 50 条）

**验收标准**：
- `get_history(max_tokens=4000)` 返回的消息总 token 数 <= 4000
- `get_history()` 无参数调用行为不变（向后兼容）
- 空会话返回空列表

#### 4.2 集成到 context builder — `nanobot/agent/context.py` + `nanobot/agent/loop.py`

**改动**：

1. 在 `AgentLoop._process_message()` 中计算历史 token 预算：
```python
if self.compressor:
    window = self.compressor.context_window
    reserve = self.compressor.reserve_ratio
    system_tokens = count_message_tokens([messages[0]])  # system prompt
    history_budget = int(window * (1 - reserve)) - system_tokens - 2000  # 2000 for current msg + memory
    history = session.get_history(max_tokens=history_budget)
else:
    history = session.get_history()
```

**验收标准**：
- 使用 64K 窗口模型时，历史消息不超过约 50K tokens
- 使用 200K 窗口模型时，可以容纳更多历史
- 不影响 `process_direct()` 的 CLI 模式

---

## 阶段五：对话摘要压缩（可选，低优先级）

### 目标
当历史消息被裁剪时，用 LLM 生成的摘要替代被丢弃的旧消息，保留对话连续性。这是成本最高的优化，仅在用户明确启用时生效。

### 修改文件

#### 5.1 对话摘要器 — `nanobot/agent/compressor.py`

**改动**：新增 `async summarize_history()` 方法：

```python
async def summarize_history(
    self,
    messages: list[dict],
    provider: LLMProvider,
    model: str | None = None,
) -> list[dict]:
    """
    将旧历史消息压缩为摘要：
    1. 保留 system prompt
    2. 将被裁剪的历史消息发送给 LLM 生成摘要
    3. 摘要作为一条 system 消息插入到 system prompt 之后
    4. 保留最近 N 轮完整对话

    使用 compression_model（如果配置了更便宜的模型）以降低成本。
    """
```

**摘要 prompt 模板**：
```
请将以下对话历史压缩为简洁的摘要，保留关键信息、用户偏好和重要决策。
用中文输出，控制在 500 字以内。

{conversation_text}
```

**验收标准**：
- 摘要生成使用配置的 `compression_model`（默认使用当前模型）
- 摘要失败时静默降级（只丢弃旧消息，不插入摘要）
- 摘要内容被安全包装（类似 memory 的 safety prefix）

#### 5.2 配置扩展 — `nanobot/config/schema.py`

**改动**：在 `AgentDefaults` 中新增：
```python
enable_history_summarization: bool = False  # 默认关闭，需要额外 API 调用
```

**验收标准**：
- 默认关闭，不产生额外 API 成本
- 开启后，长对话中旧消息被摘要替代

---

## 测试策略

### 单元测试 — `tests/test_compressor.py`（新建）

1. **token 估算测试**
   - 纯英文文本估算
   - 纯中文文本估算
   - 混合中英文估算
   - 包含 tool_calls 的 messages 估算

2. **工具结果裁剪测试**
   - 短结果不裁剪
   - 超长结果软裁剪（保留首尾）
   - 累积超限硬裁剪（旧结果优先）
   - 文件操作结果保留路径

3. **紧急压缩测试**
   - 压缩后 token 数在预算内
   - system prompt 完整保留
   - 最近消息完整保留

4. **动态历史窗口测试**
   - token 预算限制生效
   - 向后兼容（无参数调用）

### 集成测试 — `tests/test_overflow_recovery.py`（新建）

1. **溢出检测测试**
   - 各 provider 的错误格式识别
   - 非溢出错误不触发重试

2. **端到端重试测试**
   - mock LLM 第一次返回 overflow，第二次成功
   - 验证压缩后的 messages 格式正确

### 模型窗口测试 — `tests/test_model_limits.py`（新建）

1. 已知模型返回正确窗口大小
2. 未知模型返回默认值
3. 模糊匹配正确工作

---

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| token 估算不准导致仍然溢出 | 中 | 中 | 阶段三的溢出恢复作为安全网；估算值乘以 1.1 安全系数 |
| 工具结果裁剪丢失关键信息 | 高 | 低 | 文件操作保留路径；软裁剪保留首尾；用户可关闭压缩 |
| 摘要压缩增加 API 成本 | 中 | 确定 | 阶段五默认关闭；支持配置便宜模型 |
| 紧急压缩过于激进 | 中 | 低 | 保留最近 2 轮完整对话；日志记录压缩详情 |
| litellm 版本更新改变错误格式 | 低 | 中 | 错误检测使用多种模式匹配；兜底返回原始错误 |

---

## 阶段六：语义感知裁剪优先级

### 目标
当前裁剪纯粹基于时间顺序（旧的先裁），不区分消息的语义重要性。本阶段为消息引入"重要性标签"，让包含关键决策的旧消息比纯查询类结果存活更久。

### 新建文件

#### 6.1 消息重要性分类器 — `nanobot/agent/message_classifier.py`（新建）

**核心函数**：

```python
class MessagePriority(Enum):
    HIGH = "high"       # 决策、约束、TODO、用户明确指令
    NORMAL = "normal"   # 普通对话
    LOW = "low"         # 纯查询类 tool result（ls、cat、grep、search）

def classify_message(message: dict) -> MessagePriority:
    """
    基于规则的快速分类（无 LLM 调用）：
    1. tool result 按工具名分类：
       - LOW: shell(ls/cat/grep/find/head/tail), read_file, search_web
       - NORMAL: write_file, edit_file, execute_shell(非查询)
       - HIGH: 包含 "决定"/"TODO"/"注意"/"重要"/"constraint" 等关键词的结果
    2. user message：
       - HIGH: 包含决策性关键词（"请记住"/"以后都"/"规则是"/"不要"）
       - NORMAL: 其他
    3. assistant message：
       - HIGH: 包含 TODO/决策/架构描述
       - NORMAL: 其他
    """

LOW_PRIORITY_TOOLS = {
    "read_file", "search_web", "list_directory",
}
LOW_PRIORITY_SHELL_PATTERNS = [
    r"^(ls|cat|head|tail|grep|find|wc|file|stat|pwd|echo|which)\b",
]
HIGH_PRIORITY_KEYWORDS_ZH = ["决定", "TODO", "注意", "重要", "规则", "约束", "不要", "必须", "请记住"]
HIGH_PRIORITY_KEYWORDS_EN = ["decide", "todo", "important", "constraint", "must", "remember", "rule"]
```

**验收标准**：
- `classify_message({"role": "tool", "name": "read_file", ...})` 返回 `LOW`
- 包含 "请记住以后都用 TypeScript" 的 user message 返回 `HIGH`
- 分类延迟 < 1ms（纯规则，无 LLM）

#### 6.2 集成到 compressor — `nanobot/agent/compressor.py`

**改动**：修改 `compress_messages()` 的裁剪顺序：

```python
# 原来：按时间从旧到新裁剪
# 改为：按 (priority, age) 排序，LOW 优先裁剪，同优先级内旧的优先
prunable = sorted(prunable_messages, key=lambda m: (m.priority.value, m.index))
```

**验收标准**：
- 一个旧的 HIGH 优先级消息比新的 LOW 优先级消息更晚被裁剪
- 分类不影响 system prompt 和最近用户消息的保护

---

## 阶段七：扩展 Pruning 到非 Tool Result 消息

### 目标
当前阶段二只裁剪 tool result。用户粘贴的大段代码、日志、或 assistant 的超长回复同样会撑爆上下文。

### 修改文件

#### 7.1 通用消息裁剪 — `nanobot/agent/compressor.py`

**改动**：在 `compress_messages()` 中增加对 user/assistant 消息的 soft trim：

```python
# 新增配置
max_user_message_chars: int = 30000      # 单条 user message 上限
max_assistant_message_chars: int = 20000  # 单条 assistant message 上限（不含最近一条）

def trim_long_message(self, message: dict, max_chars: int) -> dict:
    """
    对超长的 user/assistant 消息做 soft trim：
    - 保留前 60% 和后 20% 的字符
    - 中间替换为 "[...已省略 {N} 字符...]"
    - 不裁剪 system 消息
    - 不裁剪最近一轮的 user 和 assistant 消息
    """
```

#### 7.2 配置扩展 — `nanobot/config/schema.py`

```python
max_user_message_chars: int = 30000
max_assistant_message_chars: int = 20000
```

**验收标准**：
- 用户粘贴 50000 字符的代码块，被裁剪到约 30000 字符（保留首尾）
- 最近一轮对话不受影响
- 配置为 0 时禁用此功能

---

## 阶段八：分层摘要与摘要缓存

### 目标
不等到 overflow 才做 compaction，而是持续维护"滚动摘要"，类似 git 的 pack 机制。

### 新建文件

#### 8.1 滚动摘要管理器 — `nanobot/agent/rolling_summary.py`（新建）

```python
class RollingSummary:
    """
    持续维护对话摘要，避免 compaction 时重复调用 LLM。

    策略：
    - 每 N 轮对话（默认 10 轮）后，对最旧的一批消息生成摘要
    - 摘要缓存在 session 元数据中（JSONL 的 metadata 行）
    - Compaction 时直接使用已有摘要，只需摘要新增部分
    - 摘要采用结构化格式（见阶段九）
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str | None = None,  # 使用便宜模型
        interval: int = 10,         # 每 10 轮触发
    ): ...

    async def maybe_summarize(
        self,
        messages: list[dict],
        session: Session,
    ) -> str | None:
        """
        检查是否需要生成新摘要：
        1. 距上次摘要已过 N 轮
        2. 取上次摘要点之后的消息
        3. 调用 LLM 生成增量摘要
        4. 与已有摘要合并
        5. 缓存到 session metadata
        """

    def get_cached_summary(self, session: Session) -> str | None:
        """获取已缓存的摘要"""
```

#### 8.2 集成到 agent loop — `nanobot/agent/loop.py`

**改动**：在每次 LLM 调用完成后（非 tool call 时），检查是否需要生成增量摘要：

```python
# 在 agent loop 的 while 循环末尾
if self.rolling_summary and not response.tool_calls:
    await self.rolling_summary.maybe_summarize(messages, session)
```

**验收标准**：
- 20 轮对话后，session metadata 中有缓存的摘要
- 紧急压缩时优先使用缓存摘要，不再调用 LLM
- 摘要缓存不影响 session JSONL 的向后兼容性（存储在 metadata 字段中）

---

## 阶段九：结构化摘要格式

### 目标
用结构化格式替代自由文本摘要，减少信息丢失，便于增量更新。

### 修改文件

#### 9.1 结构化摘要模板 — `nanobot/agent/compressor.py`

**改动**：修改摘要 prompt 和解析逻辑：

```python
STRUCTURED_SUMMARY_PROMPT = """请将以下对话历史压缩为结构化摘要，使用以下 JSON 格式：
{
  "decisions": ["已做出的决策和选择"],
  "constraints": ["用户提出的约束和规则"],
  "open_questions": ["未解决的问题"],
  "file_changes": ["已修改的文件及变更摘要"],
  "context": "对话的整体背景和进展（200字以内）"
}

对话历史：
{conversation_text}
"""

def parse_structured_summary(raw: str) -> dict:
    """解析 LLM 返回的结构化摘要，容错处理"""

def format_summary_for_context(summary: dict) -> str:
    """将结构化摘要格式化为注入上下文的文本"""
```

**验收标准**：
- 摘要包含 decisions、constraints、open_questions、file_changes、context 五个字段
- JSON 解析失败时降级为自由文本摘要
- 增量更新时可以 merge 两个结构化摘要（新的 decisions 追加到旧的后面）

---

## 实施顺序与依赖

```
阶段一（基础设施）──→ 阶段二（工具裁剪）──→ 阶段三（溢出恢复）
       │                    │                      │
       │               阶段六（语义优先级）    阶段四（动态历史）
       │               阶段七（非工具裁剪）         │
       │                                      阶段五（摘要压缩）
       │                                           │
       │                                      阶段八（滚动摘要缓存）
       │                                      阶段九（结构化摘要）
```

- 阶段一是所有后续阶段的前置依赖
- 阶段二是 ROI 最高的，建议最先完成
- 阶段六、七依赖阶段二的 compressor 框架，可并行开发
- 阶段三是安全网，建议在阶段二之后尽快完成
- 阶段四依赖阶段一的 token 计数
- 阶段五~九是增强功能，按需实施：
  - 阶段五（基础摘要）→ 阶段八（滚动缓存）→ 阶段九（结构化格式）是递进关系
  - 阶段六（语义优先级）和阶段七（非工具裁剪）相互独立

**推荐实施优先级**：一 → 二 → 三 → 四 → 七 → 六 → 五 → 八 → 九

## 文件变更总览

| 文件 | 操作 | 阶段 |
|------|------|------|
| `nanobot/storage/chunker.py` | 修改 | 一 |
| `nanobot/providers/model_limits.py` | 新建 | 一 |
| `nanobot/config/schema.py` | 修改 | 一、五 |
| `nanobot/agent/compressor.py` | 新建 | 二、三、五 |
| `nanobot/agent/loop.py` | 修改 | 二、三、四 |
| `nanobot/providers/litellm_provider.py` | 修改 | 三 |
| `nanobot/providers/base.py` | 修改 | 三（chat 签名） |
| `nanobot/session/manager.py` | 修改 | 四 |
| `nanobot/agent/context.py` | 修改 | 四 |
| `tests/test_compressor.py` | 新建 | 二、三 |
| `tests/test_overflow_recovery.py` | 新建 | 三 |
| `tests/test_model_limits.py` | 新建 | 一 |
| `nanobot/agent/message_classifier.py` | 新建 | 六 |
| `nanobot/agent/rolling_summary.py` | 新建 | 八 |
| `tests/test_message_classifier.py` | 新建 | 六 |

## 预估复杂度

- 阶段一：LOW（纯数据 + 简单函数）
- 阶段二：MEDIUM（核心压缩逻辑 + 集成）
- 阶段三：MEDIUM（错误处理 + 重试链）
- 阶段四：LOW（修改现有函数签名）
- 阶段五：MEDIUM（LLM 调用 + 异步处理）
- 阶段六：LOW（纯规则分类 + compressor 排序调整）
- 阶段七：LOW（复用阶段二的 trim 逻辑，扩展到 user/assistant）
- 阶段八：MEDIUM（增量摘要 + session metadata 缓存）
- 阶段九：LOW（prompt 模板 + JSON 解析）
- 总体：MEDIUM
