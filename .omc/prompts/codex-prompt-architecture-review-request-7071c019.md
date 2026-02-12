---
provider: "codex"
agent_role: "architect"
model: "gpt-5.3-codex"
files:
  - "D:\\workSpace\\beternanobot\\.omc\\plans\\nanobot-smart-memory-v2.md"
timestamp: "2026-02-11T09:17:53.610Z"
---

--- File: D:\workSpace\beternanobot\.omc\plans\nanobot-smart-memory-v2.md ---
# Nanobot 智能记忆系统 v2.0

> 状态: **Final Plan**
> 创建时间: 2026-02-11
> 技术栈: 纯 Python (sentence-transformers + ChromaDB)

---

## 一、方案概述

### 1.1 与 v1.0 的变化

| 维度 | v1.0 (QMD) | v2.0 (纯 Python) |
|------|------------|------------------|
| Embedding | QMD 本地模型 | sentence-transformers |
| 向量存储 | QMD sqlite-vec | ChromaDB |
| 运行时 | Bun + Node.js | **纯 Python** |
| 模型大小 | ~2GB | **~118MB** |
| 安装 | bun install | **pip install** |

### 1.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                Nanobot Smart Memory v2.0                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  写入层              存储层              检索层                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ AgentHooks  │    │ Markdown    │    │ Hybrid Search       │ │
│  │ ├─ToolObs   │───▶│ Files       │◀───│ ├─ BM25 (FTS5)      │ │
│  │ └─Summary   │    │ (不变)      │    │ └─ Vector (Chroma)  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                            │                    ▲               │
│                            ▼                    │               │
│                     ┌─────────────┐    ┌─────────────────────┐ │
│                     │ .storage/   │    │ sentence-transformers│ │
│                     │ ├─ chroma/  │    │ multilingual-MiniLM │ │
│                     │ └─ fts.db   │    │ (118MB 本地)        │ │
│                     └─────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、技术选型

### 2.1 Embedding 模型

**选择：`paraphrase-multilingual-MiniLM-L12-v2`**

| 属性 | 值 |
|------|----|
| 大小 | 118MB |
| 维度 | 384 |
| 语言 | 50+ (含中英文) |
| 下载量 | 1800万+ |

### 2.2 依赖

```bash
pip install sentence-transformers chromadb
```

---

## 三、文件结构

### 新增文件

```
nanobot/
├── agent/
│   ├── hooks.py           # Agent 生命周期 Hooks
│   ├── observation.py     # 工具观察记录
│   ├── summarizer.py      # 会话摘要生成
│   ├── memory.py          # HybridMemoryStore (重构)
│   └── retrieval.py       # 混合检索
├── storage/
│   ├── embedding.py       # Embedding 服务
│   ├── vectorstore.py     # ChromaDB 封装
│   └── fts.py             # FTS5 全文搜索
```

### 数据存储

```
workspace/
├── memory/                 # Markdown (不变)
│   ├── MEMORY.md
│   └── 2026-02-11.md
├── .storage/               # 新增：索引
│   ├── chroma/
│   └── fts.sqlite3
```

---

## 四、核心模块

### 4.1 Embedding 服务

```python
# nanobot/storage/embedding.py

from sentence_transformers import SentenceTransformer

class LocalEmbedding:
    MODELS = {
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 118MB
        "chinese": "BAAI/bge-small-zh-v1.5",  # 95MB
        "english": "all-MiniLM-L6-v2",  # 22MB
    }

    def __init__(self, model_name: str = "multilingual"):
        actual_model = self.MODELS.get(model_name, model_name)
        self.model = SentenceTransformer(actual_model)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
```

### 4.2 向量存储 (ChromaDB)

```python
# nanobot/storage/vectorstore.py

import chromadb
from dataclasses import dataclass

@dataclass
class VectorSearchResult:
    id: str
    text: str
    score: float
    metadata: dict

class VectorStore:
    def __init__(self, storage_path: Path, embedding: LocalEmbedding):
        self.embedding = embedding
        self.client = chromadb.PersistentClient(path=str(storage_path))
        self.collection = self.client.get_or_create_collection("memory")

    def add(self, text: str, metadata: dict = None) -> str:
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        embedding = self.embedding.embed(text)
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )
        return doc_id

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        query_embedding = self.embedding.embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        # 转换距离为相似度
        return [
            VectorSearchResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                score=1 / (1 + results["distances"][0][i]),
                metadata=results["metadatas"][0][i]
            )
            for i in range(len(results["ids"][0]))
        ]
```

### 4.3 全文搜索 (FTS5)

```python
# nanobot/storage/fts.py

import sqlite3

class FullTextSearch:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(id, text, title, source)
        """)

    def add(self, text: str, doc_id: str, title: str = ""):
        self.conn.execute(
            "INSERT INTO docs_fts VALUES (?, ?, ?, ?)",
            (doc_id, text, title, "")
        )
        self.conn.commit()

    def search(self, query: str, limit: int = 10):
        cursor = self.conn.execute("""
            SELECT id, text, bm25(docs_fts) as score
            FROM docs_fts WHERE docs_fts MATCH ?
            ORDER BY score LIMIT ?
        """, (query, limit))
        return cursor.fetchall()
```

### 4.4 混合检索

```python
# nanobot/agent/retrieval.py

class HybridRetrieval:
    def __init__(self, workspace: Path, embedding_model: str = "multilingual"):
        self.embedding = LocalEmbedding(embedding_model)
        self.vector_store = VectorStore(workspace / ".storage/chroma", self.embedding)
        self.fts = FullTextSearch(workspace / ".storage/fts.sqlite3")

    def search(self, query: str, limit: int = 10) -> list:
        # 1. 向量搜索
        vector_results = self.vector_store.search(query, limit * 2)

        # 2. 全文搜索
        fts_results = self.fts.search(query, limit * 2)

        # 3. RRF 融合
        return self._rrf_fusion(vector_results, fts_results, limit)

    def _rrf_fusion(self, vec_results, fts_results, limit, k=60):
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, r in enumerate(vec_results, 1):
            scores[r.id] = scores.get(r.id, 0) + 1 / (k + rank)
        for rank, r in enumerate(fts_results, 1):
            scores[r[0]] = scores.get(r[0], 0) + 1 / (k + rank)

        # 排序返回
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:limit]

    def progressive_retrieve(self, query: str):
        """三层渐进式检索"""
        results = self.search(query, limit=20)

        high = [r for r in results if r.score >= 0.5][:3]   # 获取完整内容
        medium = [r for r in results if 0.3 <= r.score < 0.5][:5]  # 只用摘要

        for r in high:
            r.is_full = True

        return high + medium
```

### 4.5 Agent Hooks

```python
# nanobot/agent/hooks.py

from abc import ABC
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ToolExecution:
    tool_name: str
    arguments: dict
    result: str
    error: Exception | None
    timestamp: datetime
    duration_ms: float
    session_key: str

class AgentHook(ABC):
    async def on_message_received(self, session_key, content): pass
    async def on_tool_executed(self, execution: ToolExecution): pass
    async def on_session_end(self, session_key, history, response): pass

class HookManager:
    def __init__(self):
        self._hooks = []

    def register(self, hook: AgentHook):
        self._hooks.append(hook)

    async def emit_tool_executed(self, execution):
        for hook in self._hooks:
            await hook.on_tool_executed(execution)
```

### 4.6 观察记录 Hook

```python
# nanobot/agent/observation.py

SIGNIFICANT_TOOLS = {"write_file", "edit_file", "exec"}

class ObservationHook(AgentHook):
    def __init__(self, memory_dir: Path, retrieval: HybridRetrieval):
        self.memory_dir = memory_dir
        self.retrieval = retrieval

    async def on_tool_executed(self, execution: ToolExecution):
        if execution.tool_name not in SIGNIFICANT_TOOLS:
            return
        if execution.error:
            return

        # 生成观察内容
        content = self._format(execution)

        # 1. 写入 Markdown
        self._write_markdown(content)

        # 2. 索引到向量库
        self.retrieval.index_document(content, title=execution.tool_name)

    def _format(self, exec):
        return f"""## 🔧 {exec.tool_name} @ {exec.timestamp:%H:%M}

**\1**: {exec.arguments}
**\1**: {exec.result[:300]}...
"""
```

### 4.7 会话摘要 Hook

```python
# nanobot/agent/summarizer.py

SUMMARY_PROMPT = """Summarize this conversation:
1. What user wanted
2. Key actions taken
3. Outcome
Keep under 200 words.

{conversation}
"""

class SummaryHook(AgentHook):
    def __init__(self, memory_dir, retrieval, provider, min_turns=3):
        self.memory_dir = memory_dir
        self.retrieval = retrieval
        self.provider = provider
        self.min_turns = min_turns

    async def on_session_end(self, session_key, history, response):
        if len([m for m in history if m["role"]=="user"]) < self.min_turns:
            return

        # 生成摘要
        summary = await self._generate_summary(history)

        # 写入 MEMORY.md
        self._save(session_key, summary)

        # 索引
        self.retrieval.index_document(summary, title=f"Session: {session_key}")

    async def _generate_summary(self, history):
        conv = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
        resp = await self.provider.chat([{"role": "user", "content": SUMMARY_PROMPT.format(conversation=conv)}])
        return resp.content
```

### 4.8 混合记忆存储

```python
# nanobot/agent/memory.py

class HybridMemoryStore:
    def __init__(self, workspace: Path, embedding_model="multilingual"):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self._retrieval = None
        self._model = embedding_model

    def initialize(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._retrieval = HybridRetrieval(self.workspace, self._model)
        self._index_existing_files()

    @property
    def retrieval(self):
        if not self._retrieval:
            self.initialize()
        return self._retrieval

    async def get_context(self, query: str = None) -> str:
        """获取记忆上下文"""
        parts = []

        # 语义检索
        if query:
            results = self.retrieval.progressive_retrieve(query)
            if results:
                parts.append(self.retrieval.format_context(results))

        # 长期记忆头部
        if (self.memory_dir / "MEMORY.md").exists():
            content = (self.memory_dir / "MEMORY.md").read_text()[-1000:]
            parts.append(f"## Long-term Memory\n{content}")

        return "\n\n---\n\n".join(parts)
```

---

## 五、AgentLoop 集成

```python
# nanobot/agent/loop.py (修改)

class AgentLoop:
    def __init__(self, ..., enable_observations=True, enable_summaries=True):
        # 初始化记忆
        self.memory = HybridMemoryStore(workspace)

        # 初始化 Hooks
        self.hooks = HookManager()

        if enable_observations:
            self.hooks.register(ObservationHook(
                workspace / "memory",
                self.memory.retrieval
            ))

        if enable_summaries:
            self.hooks.register(SummaryHook(
                workspace / "memory",
                self.memory.retrieval,
                provider
            ))

    async def run(self):
        self.memory.initialize()  # 加载 embedding 模型
        # ...

    async def _process_message(self, msg):
        # 使用语义检索构建上下文
        memory_context = await self.memory.get_context(query=msg.content)

        # ... 现有逻辑 ...

        for tool_call in response.tool_calls:
            start = time.time()
            result = await self.tools.execute(tool_call.name, tool_call.arguments)
            duration = (time.time() - start) * 1000

            # 触发 Hook
            await self.hooks.emit_tool_executed(ToolExecution(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=result,
                error=None,
                timestamp=datetime.now(),
                duration_ms=duration,
                session_key=msg.session_key
            ))

        # 会话结束
        await self.hooks.emit_session_end(
            session_key=msg.session_key,
            history=session.messages,
            response=final_content
        )
```

---

## 六、配置

```yaml
# config.yaml
memory:
  embedding_model: "multilingual"  # multilingual | chinese | english
  enable_observations: true
  enable_summaries: true
  summary_min_turns: 3
  retrieval:
    score_high: 0.5
    score_medium: 0.3
    max_full_docs: 3
```

---

## 七、安装与使用

### 安装

```bash
# 安装 nanobot 时包含 memory 功能
pip install nanobot[memory]

# 或单独安装依赖
pip install sentence-transformers chromadb
```

### 首次运行

首次运行会自动下载 embedding 模型 (~118MB)：

```
~/.cache/huggingface/hub/
└── models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
```

---

## 八、实施计划

| Phase | 内容 | 时间 |
|-------|------|------|
| 1 | 存储层 (embedding, vectorstore, fts) | 1 天 |
| 2 | 检索层 (retrieval, memory) | 1 天 |
| 3 | Hooks (hooks, observation, summarizer) | 1 天 |
| 4 | 集成 (loop 修改, 测试) | 1 天 |

**总计：4 天**

---

## 九、验收标准

- [ ] `pip install sentence-transformers chromadb` 成功
- [ ] Embedding 模型自动下载并加载
- [ ] 向量搜索返回相关结果
- [ ] 全文搜索正常工作
- [ ] 混合搜索 RRF 融合正确
- [ ] 工具观察自动记录到 Markdown
- [ ] 工具观察自动索引到向量库
- [ ] 会话摘要生成并保存
- [ ] Token 消耗降低 50%+

---

## 十、优势总结

| 维度 | 说明 |
|------|------|
| **纯 Python** | 无需 Bun/Node.js |
| **轻量模型** | 118MB vs 2GB |
| **简单安装** | pip install |
| **离线可用** | 本地 embedding |
| **中文支持** | 多语言模型 |
| **混合检索** | BM25 + 向量 |
| **自动记录** | Hooks 系统 |
| **会话摘要** | LLM 生成 |

---

*Plan Status: Final v2.0*
*Ready for Implementation*


[HEADLESS SESSION] You are running non-interactively in a headless pipeline. Produce your FULL, comprehensive analysis directly in your response. Do NOT ask for clarification or confirmation - work thoroughly with all provided context. Do NOT write brief acknowledgments - your response IS the deliverable.

# Architecture Review Request

请以架构师的视角分析 Nanobot Smart Memory v2.0 方案，重点关注：

## 分析维度

### 1. 稳健性 (Robustness)
- 错误处理和容错机制
- 数据一致性保障
- 降级策略
- 边界条件处理

### 2. 高效性 (Efficiency)
- 资源使用效率（内存、CPU）
- 延迟优化
- 并发处理
- 缓存策略

### 3. 可扩展性 (Scalability)
- 模块解耦程度
- 接口抽象层
- 未来扩展能力

### 4. 可维护性 (Maintainability)
- 代码结构清晰度
- 依赖管理
- 测试友好性

## 期望输出

1. 识别潜在问题和风险
2. 提出具体优化建议
3. 给出优先级排序
4. 如有必要，提供代码示例
