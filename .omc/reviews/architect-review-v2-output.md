这份 v2.0 方案的方向是对的：**纯 Python、本地 embedding、Hybrid Search、Hook 自动沉淀**，非常符合 Nanobot 的轻量定位。  
但从架构审查角度看，它目前更接近“可实现的技术草案”，距离“可长期稳定运行的生产方案”还有几处关键缺口，尤其是**一致性、错误隔离、检索结果契约、异步性能路径**。

下面按你要求的四个维度给出完整评估。

---

**总体判断（架构师结论）**
- 优点明显：分层思路清晰（写入/存储/检索）、技术栈统一、可离线推理、具备未来扩展入口。
- 关键风险集中在：`Hook 同步执行`、`三路写入非事务`、`检索返回结构不一致`、`首启模型与依赖不确定性`。
- 当前方案若直接上线，最可能出现的问题是：**请求时延抖动、部分索引丢失、偶发崩溃、检索质量不稳定**。
- 建议定位为：先做 `P0 稳定性修正`，再做 `P1 性能与可扩展`，最后 `P2 质量与观测性增强`。

---

**关键风险矩阵（优先级）**

| 风险点 | 影响 | 概率 | 优先级 |
|---|---|---|---|
| 检索结果契约不一致（`search` 返回 id，`progressive` 读取 score/text） | 直接功能错误 | 高 | P0 |
| Markdown/Chroma/FTS 三写入无一致性机制 | 数据丢失与索引漂移 | 高 | P0 |
| Hook 未隔离异常，可能中断主流程 | 聊天主链路稳定性下降 | 中高 | P0 |
| embedding 编码与索引在请求路径同步执行 | 高延迟、吞吐下降 | 高 | P0 |
| 模型下载/初始化失败无降级 | 首次运行不可用 | 中高 | P0 |
| FTS5 MATCH 查询边界处理不足 | 查询报错或召回异常 | 中 | P1 |
| 模块抽象偏具体实现（Chroma/FTS耦合） | 未来替换成本高 | 中 | P1 |
| 测试与可观测性未定义 | 回归与运维难 | 中 | P1 |

---

**1) 稳健性（Robustness）**

- **错误处理与容错**
  - `HookManager` 当前是串行 await，任一 Hook 抛错会影响主流程；建议统一“**主流程不被 Hook 阻断**”。
  - `SummaryHook`、`ObservationHook` 的外部依赖（LLM、向量库、文件系统）都可能失败，需要失败重试与超时策略。
  - `FTS MATCH` 对特殊字符、引号、操作符较敏感，用户 query 需 sanitize，否则会报 SQL 错误。
- **数据一致性**
  - 现在同一条记忆要写 Markdown + Chroma + FTS，属于“三写分布式事务”问题；当前没有事务边界与重放机制。
  - `VectorStore.add` 用 `md5(text)[:12]` 作为 id，重复内容易冲突，Chroma `add` 可能报重复 ID；建议 `upsert` + 业务唯一键。
- **降级策略**
  - 首次模型下载失败、磁盘不可写、sqlite/chroma异常时，应降级为“仅 Markdown 记忆”。
  - 检索不可用时，不应让 AgentLoop 失败，应返回空上下文并记录 telemetry。
- **边界条件**
  - 空 query、超长 query、二进制工具输出、超长 tool result、会话异常中断（未触发 session_end）都需要明确处理。
  - “离线可用”与“首次自动下载模型”有冲突：首次无网不可用，需要预热包或本地模型路径配置。

---

**2) 高效性（Efficiency）**

- **CPU/内存**
  - `sentence-transformers` 实际运行开销不止模型文件 118MB，含运行时（Torch）后常见内存占用会显著上升。
  - 在 async 主循环中直接 `model.encode` 会阻塞事件循环，影响并发会话。
- **延迟优化**
  - 当前写观察、索引、摘要都可能在用户请求路径执行，建议改为“**异步落盘 + 后台索引**”。
  - `MEMORY.md` 每次 `read_text()[-1000:]` 是整文件读，文件增长后会变慢；应改为 tail 读取或增量缓存。
- **并发处理**
  - SQLite 连接、Chroma 客户端并发语义需明确；至少需单写队列或锁，避免竞态。
  - Hook 可并行执行，但需限流（Semaphore）避免资源争用。
- **缓存策略**
  - Query embedding 可做 LRU/TTL 缓存；常见重复问题可直接命中。
  - 索引写入可 batch（`add_batch`），显著降低 IO 与编码开销。

---

**3) 可扩展性（Scalability）**

- **模块解耦**
  - 方案结构已经分层，但实现层仍强绑定 `SentenceTransformer + Chroma + SQLite FTS`，建议引入 Protocol/ABC 抽象。
  - 建议抽象：`EmbeddingProvider`、`VectorIndex`、`TextIndex`、`FusionStrategy`、`MemorySink`。
- **接口抽象层**
  - 检索结果应统一实体（包含 text/metadata/score/source），避免上层逻辑依赖具体后端返回结构。
  - Hook 事件模型建议固定 schema（含 event_id/session_key/tool_name/timestamps），便于后续接入更多 sink。
- **未来扩展能力**
  - 多租户/多机器人：需 namespace（collection per bot/workspace）与过滤维度（tenant/session/time）。
  - 数据生命周期：建议 retention + compaction + reindex 策略，否则长期运行会退化。

---

**4) 可维护性（Maintainability）**

- **代码结构**
  - 文件划分方向正确，但计划中的多处方法是占位（`index_document/_write_markdown/_save/emit_session_end`）且接口未闭环。
  - 建议先定义完整“最小可用契约”再铺模块，避免后续跨文件返工。
- **依赖管理**
  - `sentence-transformers`/`chromadb` 版本建议 pin 范围，避免上游破坏性变更。
  - `nanobot[memory]` extra 设计合理，但需要清晰区分 `core` 与 `memory` optional import 路径。
- **测试友好性**
  - 需建立 3 层测试：单元（融合算法/清洗逻辑）+ 集成（tmp workspace + sqlite + chroma）+ 故障注入（模型加载失败、磁盘满、Hook异常）。
  - 增加性能回归基线（QPS、P95、索引延迟），避免“功能增加但体验退化”。
- **可观测性**
  - 必须加入 metrics：`index_success_rate`、`index_lag_ms`、`retrieval_p95_ms`、`fallback_count`、`hook_error_count`。

---

**你这版方案中最需要立即修正的具体问题（代码层面）**

- `HybridRetrieval.search()` 返回 `sorted_ids`，但 `progressive_retrieve()` 用 `r.score`、`r.is_full`，对象契约冲突。
- `HookManager` 只示例了 `emit_tool_executed`，但 loop 使用了 `emit_session_end`，接口不完整。
- `VectorStore.add()` 使用固定 hash id 且 `add` 非 `upsert`，重复文档会失败或冲突。
- 观察/摘要落地与索引顺序未定义，失败重试机制缺失，数据可能“写了 Markdown 但没索引”。
- FTS 表结构无 `created_at/session_key/type`，后续检索过滤与治理能力不足。

---

**优化建议（按优先级）**

1. **P0（必须先做，建议 1-2 天内）**
- 修复检索返回契约：统一返回 `RetrievedDoc`，包含 `text/metadata/rrf_score/vector_score/bm25_score`。
- Hook 失败隔离：任何 Hook 错误只记日志，不影响主链路响应。
- 引入“单一写入事实源”（outbox/journal）：先写事件，再异步 fan-out 到 Markdown/Chroma/FTS，支持重放。
- embedding/索引移出请求主路径：放后台 worker，主流程只 enqueue。
- 实现降级开关：memory backend 不可用时自动 fallback 到仅 Markdown/无记忆模式。

2. **P1（本迭代可做，建议 3-5 天）**
- FTS query sanitize + 中文分词策略（至少 query normalize；条件允许可引入 jieba 分词）。
- 加缓存与批处理（embedding LRU、批量 add、定时 flush）。
- 配置模型化（Pydantic）：阈值/队列长度/超时/重试都可配，并有默认安全值。
- 增加 observability 与健康检查（startup self-check: model, fts5, storage path, chroma）。

3. **P2（增强项，后续迭代）**
- 插件化后端接口（可替换 Milvus/pgvector/Whoosh 等）。
- 可选 reranker（cross-encoder）提升 top-k 精度。
- 数据生命周期管理（TTL、按会话归档、重建索引命令）。
- 质量评估集（中文/英文混合查询）与自动化检索回归。

---

**建议代码示例 1：安全 Hook 触发（不阻断主流程）**

```python
import asyncio
from loguru import logger

class HookManager:
    def __init__(self):
        self._hooks = []

    def register(self, hook):
        self._hooks.append(hook)

    async def _safe_call(self, hook, method_name, *args, **kwargs):
        method = getattr(hook, method_name, None)
        if method is None:
            return
        try:
            await method(*args, **kwargs)
        except Exception:
            logger.exception("hook failed: {}.{}", hook.__class__.__name__, method_name)

    async def emit_tool_executed(self, execution):
        await asyncio.gather(*[
            self._safe_call(h, "on_tool_executed", execution) for h in self._hooks
        ])

    async def emit_session_end(self, session_key, history, response):
        await asyncio.gather(*[
            self._safe_call(h, "on_session_end", session_key, history, response) for h in self._hooks
        ])
```

---

**建议代码示例 2：统一检索结果契约（修复当前 search/progressive 不一致）**

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrievedDoc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0

def progressive_pick(results: list[RetrievedDoc], max_full=3, max_summary=5):
    ranked = sorted(results, key=lambda r: r.rrf_score, reverse=True)
    full = ranked[:max_full]
    summary = ranked[max_full:max_full + max_summary]
    return full, summary
```

---

**建议代码示例 3：写入一致性最小实现（Outbox 思路）**

```python
# 先 append event 到 sqlite(outbox)，再由后台 worker 消费
# 每个 sink 记录 last_event_id，失败可重试，进程重启可续跑
# 这样不会出现“主流程成功但索引丢失不可追”的情况
```

（这里我给的是架构模式，具体实现建议放在 `nanobot/storage/outbox.py`，避免把复杂度散在 Hooks 内。）

---

**补充：验收标准建议升级**

你当前验收项偏“功能能跑通”，建议新增“工程可靠性”指标：
- `index_success_rate >= 99.9%`
- `retrieval_p95_ms` 在目标硬件下可控（例如 < 300ms）
- `fallback_count` 可观测且可解释
- 故障注入场景下：主流程可持续响应、数据可重放恢复
- 检索质量基线（固定评测集 top-k 命中率）

---

**最终建议（可执行排序）**
1. 先修“契约 + 隔离 + 一致性”（P0），这是能否上线的分水岭。  
2. 再做“异步化 + 缓存 + 查询清洗”（P1），这是用户体感是否顺滑的关键。  
3. 最后做“抽象层与评测体系”（P2），这是长期演进成本的关键。  

如果按这个顺序推进，v2.0 可以从“技术草案”稳定进化为“可持续运行的 memory 子系统”，并且不会破坏 Nanobot 轻量核心。