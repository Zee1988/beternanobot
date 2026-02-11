# Nanobot 智能记忆系统 v1.0

> 状态: **Final Plan**
> 创建时间: 2026-02-11
> 基于: QMD 集成 + Claude-Mem 理念借鉴

---

## 一、方案概述

### 1.1 目标

将 nanobot 的记忆系统从「简单文件存储」升级为「智能语义检索系统」，同时保持架构简洁和向后兼容。

### 1.2 核心能力

| 能力 | 现状 | 目标 |
|------|------|------|
| 存储 | Markdown 文件 | Markdown 文件（不变） |
| 检索 | 全量加载 7 天 | 语义检索 + 3 层渐进披露 |
| 观察 | 手动记录 | 自动记录重要工具调用 |
| 摘要 | 无 | 会话结束自动生成 |
| Token | 高消耗 | 降低 60-80% |

### 1.3 技术选型

```
┌─────────────────────────────────────────────────────────────────┐
│                    Nanobot Smart Memory v1                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  写入层     │  │  存储层     │  │  检索层     │             │
│  │             │  │             │  │             │             │
│  │ AgentHooks  │  │ Markdown    │  │ QMD         │             │
│  │ ├─ToolObs   │──│ Files       │──│ ├─ BM25     │             │
│  │ └─Summary   │  │             │  │ ├─ Vector   │             │
│  │             │  │ workspace/  │  │ └─ Rerank   │             │
│  └─────────────┘  │ memory/     │  └─────────────┘             │
│                   └─────────────┘                               │
│                          │                                      │
│                          ▼                                      │
│              ┌─────────────────────┐                           │
│              │  3-Layer Retrieval  │                           │
│              │  ├─ Index (compact) │                           │
│              │  ├─ Filter (score)  │                           │
│              │  └─ Detail (full)   │                           │
│              └─────────────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、设计约束

| 约束 | 决策 | 理由 |
|------|------|------|
| QMD 依赖 | 必须 | 核心检索能力依赖 QMD |
| 观察粒度 | 只记录重要工具 | 减少噪音，节省存储 |
| 会话摘要 | 需要 | 长期记忆关键功能 |
| 文件结构 | 完全保留 | 零迁移成本 |
| 平台优先级 | macOS/Linux 优先 | QMD 在这些平台更成熟 |
| 检索复杂度 | 完整 3 层 | 最大化 token 节省 |
| Web UI | 未来再加 | 聚焦核心功能 |

---

## 三、文件结构

### 3.1 新增文件

```
nanobot/
├── agent/
│   ├── hooks.py           # 新增：Agent 生命周期 Hooks
│   ├── observation.py     # 新增：观察记录系统
│   ├── summarizer.py      # 新增：会话摘要生成
│   ├── memory.py          # 重构：HybridMemoryStore
│   ├── retrieval.py       # 新增：3 层渐进检索
│   └── loop.py            # 修改：集成 Hooks
├── utils/
│   └── qmd.py             # 新增：QMD CLI 封装
└── config/
    └── schema.py          # 修改：添加 memory 配置
```

### 3.2 工作区结构（保持不变）

```
workspace/
├── memory/
│   ├── MEMORY.md              # 长期记忆（含会话摘要）
│   ├── 2026-02-11.md          # 每日笔记 + 工具观察
│   ├── 2026-02-10.md
│   └── ...
├── AGENTS.md
├── SOUL.md
└── ...
```

---

## 四、核心模块设计

### 4.1 Agent Hooks 系统

```python
# nanobot/agent/hooks.py

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass
class ToolExecution:
    """工具执行记录"""
    tool_name: str
    arguments: dict[str, Any]
    result: str
    error: Exception | None
    timestamp: datetime
    duration_ms: float
    session_key: str
    channel: str
    chat_id: str


class AgentHook(ABC):
    """
    Agent 生命周期 Hook 基类

    实现类似 Claude-Mem 的 Lifecycle Hooks，但内置于 Python。
    """

    async def on_message_received(
        self,
        session_key: str,
        channel: str,
        chat_id: str,
        content: str
    ) -> None:
        """用户消息接收时调用"""
        pass

    async def on_tool_executed(self, execution: ToolExecution) -> None:
        """工具执行后调用"""
        pass

    async def on_response_ready(
        self,
        session_key: str,
        response: str
    ) -> None:
        """Agent 响应生成后调用"""
        pass

    async def on_session_end(
        self,
        session_key: str,
        history: list[dict],
        final_response: str
    ) -> None:
        """会话结束时调用"""
        pass


class HookManager:
    """Hook 管理器"""

    def __init__(self):
        self._hooks: list[AgentHook] = []

    def register(self, hook: AgentHook) -> None:
        self._hooks.append(hook)

    async def emit_message_received(self, **kwargs) -> None:
        for hook in self._hooks:
            try:
                await hook.on_message_received(**kwargs)
            except Exception as e:
                logger.warning(f"Hook error in on_message_received: {e}")

    async def emit_tool_executed(self, execution: ToolExecution) -> None:
        for hook in self._hooks:
            try:
                await hook.on_tool_executed(execution)
            except Exception as e:
                logger.warning(f"Hook error in on_tool_executed: {e}")

    async def emit_response_ready(self, **kwargs) -> None:
        for hook in self._hooks:
            try:
                await hook.on_response_ready(**kwargs)
            except Exception as e:
                logger.warning(f"Hook error in on_response_ready: {e}")

    async def emit_session_end(self, **kwargs) -> None:
        for hook in self._hooks:
            try:
                await hook.on_session_end(**kwargs)
            except Exception as e:
                logger.warning(f"Hook error in on_session_end: {e}")
```

### 4.2 观察记录系统

```python
# nanobot/agent/observation.py

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.agent.hooks import AgentHook, ToolExecution


# 重要工具列表（只记录这些）
SIGNIFICANT_TOOLS = {
    "write_file",
    "edit_file",
    "exec",
    "read_file",  # 读取大文件时可能有价值
}


class ObservationHook(AgentHook):
    """
    工具观察记录 Hook

    借鉴 Claude-Mem 的 PostToolUse 机制，自动记录重要工具调用。
    观察记录写入当日 Markdown 文件，供 QMD 索引。
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self._today_observations: list[ToolExecution] = []

    async def on_tool_executed(self, execution: ToolExecution) -> None:
        """记录重要工具调用"""
        if execution.tool_name not in SIGNIFICANT_TOOLS:
            return

        if execution.error:
            return  # 不记录失败的调用

        self._today_observations.append(execution)

        # 写入持久化
        await self._persist_observation(execution)

        logger.debug(f"Observation recorded: {execution.tool_name}")

    async def _persist_observation(self, exec: ToolExecution) -> None:
        """写入观察到今日笔记"""
        today_file = self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

        # 生成观察内容
        content = self._format_observation(exec)

        # 追加到文件
        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            new_content = existing.rstrip() + "\n\n" + content
        else:
            header = f"# {datetime.now().strftime('%Y-%m-%d')}\n\n"
            new_content = header + content

        today_file.write_text(new_content, encoding="utf-8")

    def _format_observation(self, exec: ToolExecution) -> str:
        """格式化观察记录"""
        time_str = exec.timestamp.strftime('%H:%M:%S')
        duration = f"{exec.duration_ms:.0f}ms"

        # 根据工具类型生成标题
        title = self._generate_title(exec)

        # 截断过长的结果
        result_preview = exec.result[:300]
        if len(exec.result) > 300:
            result_preview += "... (truncated)"

        return f"""## 🔧 {title}

**Time:** {time_str} | **Duration:** {duration} | **Tool:** `{exec.tool_name}`

<details>
<summary>Arguments</summary>

```json
{json.dumps(exec.arguments, indent=2, ensure_ascii=False)[:500]}
```
</details>

**Result:**
```
{result_preview}
```
"""

    def _generate_title(self, exec: ToolExecution) -> str:
        """根据工具调用生成可读标题"""
        args = exec.arguments

        if exec.tool_name == "write_file":
            path = args.get("path", "unknown")
            return f"Created/Updated {Path(path).name}"

        elif exec.tool_name == "edit_file":
            path = args.get("path", "unknown")
            return f"Edited {Path(path).name}"

        elif exec.tool_name == "exec":
            cmd = args.get("command", "")[:50]
            return f"Executed: {cmd}"

        elif exec.tool_name == "read_file":
            path = args.get("path", "unknown")
            return f"Read {Path(path).name}"

        return f"Tool: {exec.tool_name}"
```

### 4.3 会话摘要生成器

```python
# nanobot/agent/summarizer.py

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.hooks import AgentHook
from nanobot.providers.base import LLMProvider


SUMMARY_PROMPT = """Summarize this conversation session concisely. Focus on:
1. What the user wanted to accomplish
2. Key actions taken (files created/modified, commands run)
3. Final outcome or status
4. Any important context for future sessions

Keep the summary under 200 words. Use bullet points.

Conversation:
{conversation}
"""


class SummaryHook(AgentHook):
    """
    会话摘要 Hook

    在会话结束时使用 LLM 生成摘要，追加到 MEMORY.md。
    """

    def __init__(
        self,
        memory_dir: Path,
        provider: LLMProvider,
        model: str | None = None,
        min_turns: int = 3  # 至少 3 轮对话才生成摘要
    ):
        self.memory_dir = memory_dir
        self.provider = provider
        self.model = model
        self.min_turns = min_turns
        self.memory_file = memory_dir / "MEMORY.md"

    async def on_session_end(
        self,
        session_key: str,
        history: list[dict],
        final_response: str
    ) -> None:
        """会话结束时生成摘要"""
        # 过滤太短的会话
        user_turns = sum(1 for m in history if m.get("role") == "user")
        if user_turns < self.min_turns:
            logger.debug(f"Session too short ({user_turns} turns), skipping summary")
            return

        try:
            summary = await self._generate_summary(history)
            await self._save_summary(session_key, summary)
            logger.info(f"Session summary saved for {session_key}")
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

    async def _generate_summary(self, history: list[dict]) -> str:
        """使用 LLM 生成摘要"""
        # 格式化对话历史
        conversation = "\n".join(
            f"{m['role'].upper()}: {m.get('content', '')[:500]}"
            for m in history
            if m.get("role") in ("user", "assistant")
        )

        prompt = SUMMARY_PROMPT.format(conversation=conversation)

        response = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )

        return response.content

    async def _save_summary(self, session_key: str, summary: str) -> None:
        """保存摘要到 MEMORY.md"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        entry = f"""\n---\n\n### Session: {session_key} ({timestamp})\n\n{summary}\n"""

        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")
            new_content = existing.rstrip() + entry
        else:
            new_content = f"# Long-term Memory\n{entry}"

        self.memory_file.write_text(new_content, encoding="utf-8")
```

### 4.4 QMD 封装

```python
# nanobot/utils/qmd.py

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class SearchResult:
    """QMD 搜索结果"""
    docid: str
    path: str
    title: str
    score: float
    snippet: str


class QMDClient:
    """
    QMD CLI 封装

    提供 Python 接口调用 qmd 命令行工具。
    """

    def __init__(self, collection_name: str = "memory"):
        self.collection_name = collection_name
        self._available: bool | None = None

    def is_available(self) -> bool:
        """检查 qmd 是否可用"""
        if self._available is None:
            self._available = shutil.which("qmd") is not None
        return self._available

    def require_available(self) -> None:
        """要求 qmd 必须可用，否则抛出异常"""
        if not self.is_available():
            raise RuntimeError(
                "QMD is required but not installed. "
                "Install with: bun install -g github:tobi/qmd"
            )

    async def _run(self, *args: str) -> str:
        """执行 qmd 命令"""
        self.require_available()

        proc = await asyncio.create_subprocess_exec(
            "qmd", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            raise RuntimeError(f"qmd {args[0]} failed: {error_msg}")

        return stdout.decode()

    async def init_collection(self, memory_dir: Path) -> None:
        """
        初始化 memory 集合

        Args:
            memory_dir: workspace/memory 目录路径
        """
        # 检查集合是否已存在
        try:
            result = await self._run("collection", "list", "--json")
            collections = json.loads(result)

            if any(c["name"] == self.collection_name for c in collections):
                logger.debug(f"QMD collection '{self.collection_name}' already exists")
                return
        except Exception:
            pass

        # 创建集合
        await self._run(
            "collection", "add", str(memory_dir),
            "--name", self.collection_name,
            "--mask", "**/*.md"
        )

        # 添加上下文描述
        await self._run(
            "context", "add", f"qmd://{self.collection_name}",
            "Agent memory: long-term facts, session summaries, and tool observations"
        )

        # 生成初始嵌入
        await self._run("embed")

        logger.info(f"QMD collection '{self.collection_name}' initialized")

    async def update_index(self) -> None:
        """更新索引（增量）"""
        await self._run("update")

    async def search(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.2
    ) -> list[SearchResult]:
        """
        BM25 全文搜索（Layer 1: Index）

        返回紧凑的索引结果，用于第一层筛选。
        """
        result = await self._run(
            "search", query,
            "-n", str(limit),
            "--min-score", str(min_score),
            "-c", self.collection_name,
            "--json"
        )

        docs = json.loads(result)
        return [self._parse_result(d) for d in docs]

    async def vsearch(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.2
    ) -> list[SearchResult]:
        """
        向量语义搜索（Layer 1: Index）
        """
        result = await self._run(
            "vsearch", query,
            "-n", str(limit),
            "--min-score", str(min_score),
            "-c", self.collection_name,
            "--json"
        )

        docs = json.loads(result)
        return [self._parse_result(d) for d in docs]

    async def deep_search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.3
    ) -> list[SearchResult]:
        """
        混合搜索 + 重排序（Layer 2/3: Full quality）

        使用 QMD 的 query 命令，包含：
        - 查询扩展
        - BM25 + 向量混合
        - LLM 重排序
        """
        result = await self._run(
            "query", query,
            "-n", str(limit),
            "--min-score", str(min_score),
            "-c", self.collection_name,
            "--json"
        )

        docs = json.loads(result)
        return [self._parse_result(d) for d in docs]

    async def get_document(self, path_or_docid: str) -> str:
        """
        获取完整文档内容（Layer 3: Detail）
        """
        result = await self._run(
            "get", path_or_docid,
            "--full"
        )
        return result

    def _parse_result(self, doc: dict) -> SearchResult:
        """解析 qmd JSON 结果"""
        return SearchResult(
            docid=doc.get("docid", ""),
            path=doc.get("path", ""),
            title=doc.get("title", ""),
            score=doc.get("score", 0.0),
            snippet=doc.get("snippet", "")
        )
```

### 4.5 三层渐进式检索

```python
# nanobot/agent/retrieval.py

from dataclasses import dataclass
from typing import Any

from loguru import logger

from nanobot.utils.qmd import QMDClient, SearchResult


@dataclass
class RetrievalResult:
    """检索结果"""
    docid: str
    path: str
    title: str
    score: float
    content: str  # 可能是 snippet 或 full content
    is_full: bool


class ProgressiveRetrieval:
    """
    三层渐进式检索

    借鉴 Claude-Mem 的 Progressive Disclosure 理念：
    1. Layer 1 (Index): 获取紧凑索引 (~50-100 tokens/result)
    2. Layer 2 (Filter): 按分数筛选相关结果
    3. Layer 3 (Detail): 仅获取高相关结果的完整内容

    这样可以节省 ~10x tokens。
    """

    def __init__(
        self,
        qmd: QMDClient,
        score_threshold_high: float = 0.6,   # 高相关阈值
        score_threshold_medium: float = 0.4, # 中等相关阈值
        max_full_docs: int = 3,              # 最多获取完整内容的文档数
        max_snippet_docs: int = 5,           # 最多返回摘要的文档数
    ):
        self.qmd = qmd
        self.score_threshold_high = score_threshold_high
        self.score_threshold_medium = score_threshold_medium
        self.max_full_docs = max_full_docs
        self.max_snippet_docs = max_snippet_docs

    async def retrieve(
        self,
        query: str,
        use_deep_search: bool = True
    ) -> list[RetrievalResult]:
        """
        执行三层渐进式检索

        Args:
            query: 用户查询
            use_deep_search: 是否使用混合搜索+重排序

        Returns:
            检索结果列表，高相关的包含完整内容
        """
        # Layer 1: 获取索引
        if use_deep_search:
            index_results = await self.qmd.deep_search(
                query, limit=20, min_score=0.2
            )
        else:
            index_results = await self.qmd.search(
                query, limit=20, min_score=0.2
            )

        if not index_results:
            return []

        # Layer 2: 按分数分类
        high_relevance = [
            r for r in index_results
            if r.score >= self.score_threshold_high
        ][:self.max_full_docs]

        medium_relevance = [
            r for r in index_results
            if self.score_threshold_medium <= r.score < self.score_threshold_high
        ][:self.max_snippet_docs - len(high_relevance)]

        results = []

        # Layer 3: 获取高相关文档的完整内容
        for r in high_relevance:
            try:
                full_content = await self.qmd.get_document(r.docid or r.path)
                results.append(RetrievalResult(
                    docid=r.docid,
                    path=r.path,
                    title=r.title,
                    score=r.score,
                    content=full_content,
                    is_full=True
                ))
            except Exception as e:
                logger.warning(f"Failed to get full doc {r.path}: {e}")
                # 降级使用 snippet
                results.append(self._from_search_result(r, is_full=False))

        # 中等相关只用 snippet
        for r in medium_relevance:
            results.append(self._from_search_result(r, is_full=False))

        return results

    def _from_search_result(self, r: SearchResult, is_full: bool) -> RetrievalResult:
        return RetrievalResult(
            docid=r.docid,
            path=r.path,
            title=r.title,
            score=r.score,
            content=r.snippet,
            is_full=is_full
        )

    def format_context(self, results: list[RetrievalResult]) -> str:
        """
        格式化检索结果为 context 字符串
        """
        if not results:
            return ""

        parts = []

        for r in results:
            score_pct = f"{r.score:.0%}"
            content_type = "[full]" if r.is_full else "[snippet]"

            parts.append(
                f"### {r.title} (score: {score_pct}) {content_type}\n"
                f"*Source: {r.path}*\n\n"
                f"{r.content}"
            )

        return "## Relevant Memories\n\n" + "\n\n---\n\n".join(parts)
```

### 4.6 混合记忆存储（重构）

```python
# nanobot/agent/memory.py

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger

from nanobot.utils.qmd import QMDClient
from nanobot.agent.retrieval import ProgressiveRetrieval


class HybridMemoryStore:
    """
    混合记忆存储

    整合：
    - 原有的 Markdown 文件存储
    - QMD 语义检索
    - 三层渐进式披露
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_file = self.memory_dir / "MEMORY.md"

        # QMD 客户端
        self.qmd = QMDClient(collection_name="memory")
        self.retrieval = ProgressiveRetrieval(self.qmd)

        self._initialized = False

    async def initialize(self) -> None:
        """初始化记忆系统"""
        if self._initialized:
            return

        # 确保目录存在
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 QMD
        self.qmd.require_available()
        await self.qmd.init_collection(self.memory_dir)

        self._initialized = True
        logger.info("HybridMemoryStore initialized")

    async def get_context(
        self,
        query: Optional[str] = None,
        include_long_term: bool = True,
        include_today: bool = True
    ) -> str:
        """
        获取记忆上下文

        Args:
            query: 用户查询（用于语义检索）
            include_long_term: 是否包含长期记忆头部
            include_today: 是否包含今日笔记头部

        Returns:
            格式化的记忆上下文
        """
        parts = []

        # 语义检索相关记忆
        if query:
            try:
                results = await self.retrieval.retrieve(query)
                if results:
                    context = self.retrieval.format_context(results)
                    parts.append(context)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")

        # 长期记忆头部（最新的摘要）
        if include_long_term and self.memory_file.exists():
            long_term = self._get_long_term_header()
            if long_term:
                parts.append(f"## Long-term Memory (Recent)\n\n{long_term}")

        # 今日笔记头部
        if include_today:
            today = self._get_today_header()
            if today:
                parts.append(f"## Today's Notes (Recent)\n\n{today}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def _get_long_term_header(self, max_chars: int = 1000) -> str:
        """获取长期记忆的最新部分"""
        content = self.memory_file.read_text(encoding="utf-8")

        # 取最后 max_chars 字符（最新的内容在后面）
        if len(content) > max_chars:
            # 找到一个合适的分隔点
            truncated = content[-max_chars:]
            # 从第一个 "---" 或 "###" 开始
            for marker in ["---", "###"]:
                idx = truncated.find(marker)
                if idx > 0:
                    return truncated[idx:]
            return "..." + truncated

        return content

    def _get_today_header(self, max_chars: int = 800) -> str:
        """获取今日笔记的最新部分"""
        today_file = self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

        if not today_file.exists():
            return ""

        content = today_file.read_text(encoding="utf-8")

        if len(content) > max_chars:
            # 取最后部分
            truncated = content[-max_chars:]
            idx = truncated.find("## ")
            if idx > 0:
                return truncated[idx:]
            return "..." + truncated

        return content

    async def update_index(self) -> None:
        """更新 QMD 索引"""
        try:
            await self.qmd.update_index()
        except Exception as e:
            logger.warning(f"Failed to update QMD index: {e}")

    # === 保留原有方法以保持兼容 ===

    def get_today_file(self) -> Path:
        return self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

    def read_today(self) -> str:
        today_file = self.get_today_file()
        return today_file.read_text(encoding="utf-8") if today_file.exists() else ""

    async def append_today(self, content: str) -> None:
        """追加今日笔记"""
        today_file = self.get_today_file()
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            new_content = existing.rstrip() + "\n\n" + content
        else:
            header = f"# {datetime.now().strftime('%Y-%m-%d')}\n\n"
            new_content = header + content

        today_file.write_text(new_content, encoding="utf-8")

        # 后台更新索引
        asyncio.create_task(self.update_index())

    def read_long_term(self) -> str:
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    async def write_long_term(self, content: str) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(content, encoding="utf-8")
        asyncio.create_task(self.update_index())
```

### 4.7 AgentLoop 集成

```python
# nanobot/agent/loop.py (修改部分)

import time
from nanobot.agent.hooks import HookManager, ToolExecution
from nanobot.agent.observation import ObservationHook
from nanobot.agent.summarizer import SummaryHook
from nanobot.agent.memory import HybridMemoryStore


class AgentLoop:
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        # ... 其他参数 ...
        enable_observations: bool = True,
        enable_summaries: bool = True,
    ):
        # ... 现有初始化 ...

        # 使用新的混合记忆
        self.memory = HybridMemoryStore(workspace)

        # Hook 管理器
        self.hooks = HookManager()

        # 注册默认 Hooks
        if enable_observations:
            self.hooks.register(ObservationHook(workspace / "memory"))

        if enable_summaries:
            self.hooks.register(SummaryHook(
                memory_dir=workspace / "memory",
                provider=provider,
                model=self.model
            ))

    async def run(self) -> None:
        # 初始化记忆系统
        await self.memory.initialize()

        # ... 现有 run 逻辑 ...

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        # Hook: 消息接收
        await self.hooks.emit_message_received(
            session_key=msg.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=msg.content
        )

        session = self.sessions.get_or_create(msg.session_key)

        # 使用语义检索构建上下文
        memory_context = await self.memory.get_context(query=msg.content)

        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            memory_context=memory_context,  # 新参数
            # ... 其他参数 ...
        )

        # ... 现有 agent 循环 ...

        while iteration < self.max_iterations:
            response = await self.provider.chat(...)

            if response.has_tool_calls:
                for tool_call in response.tool_calls:
                    start_time = time.time()

                    result = await self.tools.execute(
                        tool_call.name,
                        tool_call.arguments
                    )

                    duration_ms = (time.time() - start_time) * 1000

                    # Hook: 工具执行后
                    await self.hooks.emit_tool_executed(ToolExecution(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        result=result,
                        error=None,
                        timestamp=datetime.now(),
                        duration_ms=duration_ms,
                        session_key=msg.session_key,
                        channel=msg.channel,
                        chat_id=msg.chat_id
                    ))

                    # ... 添加 tool result ...
            else:
                final_content = response.content
                break

        # Hook: 响应就绪
        await self.hooks.emit_response_ready(
            session_key=msg.session_key,
            response=final_content
        )

        # 保存会话
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        # Hook: 会话结束
        await self.hooks.emit_session_end(
            session_key=msg.session_key,
            history=session.messages,
            final_response=final_content
        )

        return OutboundMessage(...)
```

---

## 五、配置扩展

```python
# nanobot/config/schema.py (添加)

@dataclass
class MemoryConfig:
    """记忆系统配置"""

    # QMD 配置
    qmd_collection_name: str = "memory"

    # 观察配置
    enable_observations: bool = True
    observation_tools: list[str] = field(default_factory=lambda: [
        "write_file", "edit_file", "exec", "read_file"
    ])

    # 摘要配置
    enable_summaries: bool = True
    summary_min_turns: int = 3
    summary_model: str | None = None  # None = 使用默认模型

    # 检索配置
    retrieval_score_high: float = 0.6
    retrieval_score_medium: float = 0.4
    retrieval_max_full_docs: int = 3
    retrieval_max_snippet_docs: int = 5
```

```yaml
# config.yaml 示例

memory:
  qmd_collection_name: memory

  # 观察记录
  enable_observations: true
  observation_tools:
    - write_file
    - edit_file
    - exec

  # 会话摘要
  enable_summaries: true
  summary_min_turns: 3

  # 检索阈值
  retrieval_score_high: 0.6
  retrieval_score_medium: 0.4
  retrieval_max_full_docs: 3
```

---

## 六、安装与初始化

### 6.1 依赖安装

```bash
# 1. 安装 Bun（如果没有）
curl -fsSL https://bun.sh/install | bash

# 2. 安装 QMD
bun install -g github:tobi/qmd

# 3. 验证安装
qmd --version
```

### 6.2 首次使用

```python
# nanobot 启动时自动检测
from nanobot.utils.qmd import QMDClient

qmd = QMDClient()
if not qmd.is_available():
    print("⚠️  QMD not installed. Memory search will not work.")
    print("   Install: bun install -g github:tobi/qmd")
    sys.exit(1)

# 自动初始化集合
await qmd.init_collection(workspace / "memory")
```

---

## 七、实施计划

### Phase 1: 基础设施（1-2 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 创建 QMD 封装 | `nanobot/utils/qmd.py` | P0 |
| 创建 Hook 系统 | `nanobot/agent/hooks.py` | P0 |
| 创建观察记录 | `nanobot/agent/observation.py` | P0 |
| 测试 QMD 集成 | `tests/test_qmd.py` | P0 |

### Phase 2: 记忆系统（1-2 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 创建渐进检索 | `nanobot/agent/retrieval.py` | P0 |
| 重构记忆存储 | `nanobot/agent/memory.py` | P0 |
| 创建会话摘要 | `nanobot/agent/summarizer.py` | P1 |
| 更新配置模式 | `nanobot/config/schema.py` | P1 |

### Phase 3: 集成与测试（1-2 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 修改 AgentLoop | `nanobot/agent/loop.py` | P0 |
| 修改 ContextBuilder | `nanobot/agent/context.py` | P0 |
| 集成测试 | `tests/test_memory_integration.py` | P0 |
| 更新文档 | `README.md`, `AGENTS.md` | P1 |

### Phase 4: 优化与完善（1 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 性能测试 | - | P1 |
| 错误处理完善 | 各文件 | P1 |
| 安装脚本 | `scripts/setup_memory.sh` | P2 |

---

## 八、验收标准

### 功能验收

- [ ] QMD 能正确索引 workspace/memory 目录
- [ ] 语义搜索返回相关结果
- [ ] 工具调用自动记录到每日笔记
- [ ] 会话结束生成摘要到 MEMORY.md
- [ ] 三层检索正确工作
- [ ] Token 消耗降低 50%+

### 兼容性验收

- [ ] 现有 memory 文件无需迁移
- [ ] 无 QMD 时给出明确错误提示
- [ ] macOS 和 Linux 正常工作

### 质量验收

- [ ] 单元测试覆盖核心逻辑
- [ ] 无 Python 类型错误
- [ ] 日志输出清晰

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| QMD 安装失败 | 中 | 高 | 提供详细安装文档和诊断脚本 |
| 模型下载慢 | 中 | 中 | 文档说明首次使用需等待 |
| Windows 兼容问题 | 中 | 中 | 文档标注平台限制 |
| 检索质量不佳 | 低 | 中 | 可调节阈值参数 |
| 摘要质量不稳定 | 中 | 低 | 可配置关闭摘要功能 |

---

## 十、未来扩展

### v1.1
- MCP HTTP daemon 模式（避免冷启动）
- 观察记录的隐私过滤（<private> 标签）

### v1.2
- Web UI 查看记忆和观察
- 记忆压缩（合并旧的每日笔记）

### v2.0
- 多 workspace 支持
- 跨会话记忆共享

---

*Plan Status: Final*
*Ready for Implementation*
