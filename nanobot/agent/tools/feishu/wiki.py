"""Feishu Wiki (knowledge base) tools."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

try:
    from lark_oapi import wiki
    from lark_oapi.api.wiki.v2 import (
        ListSpaceRequest,
        SearchNodeRequest,
    )
    FEISHU_WIKI_AVAILABLE = True
except ImportError:
    FEISHU_WIKI_AVAILABLE = False
    wiki = None


class FeishuWikiListSpaces(Tool):
    """Tool to list all accessible wiki knowledge spaces."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_wiki_list_spaces"

    @property
    def description(self) -> str:
        return "List all accessible Feishu Wiki knowledge spaces (folders)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    async def execute(self, **kwargs: Any) -> str:
        if not FEISHU_WIKI_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            request = ListSpaceRequest.builder() \
                .build()

            response = self._client.wiki.v2.space.list(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            spaces = response.data.get('spaces', []) if response.data else []
            if not spaces:
                return "No wiki spaces found"

            results = []
            for space in spaces:
                name = space.get('name', 'Unnamed')
                space_id = space.get('space_id', '')
                results.append(f"- {name} (ID: {space_id})")

            return "Wiki Spaces:\n" + "\n".join(results)
        except Exception as e:
            logger.error(f"Error listing wiki spaces: {e}")
            return f"Error: {str(e)}"


class FeishuWikiSearch(Tool):
    """Tool to search wiki knowledge base nodes."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_wiki_search"

    @property
    def description(self) -> str:
        return "Search for documents in Feishu Wiki knowledge base. Returns matching document titles and tokens."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword"
                },
                "space_id": {
                    "type": "string",
                    "description": "Specific space ID to search in (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, space_id: str = "", limit: int = 10, **kwargs: Any) -> str:
        if not FEISHU_WIKI_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            builder = SearchNodeRequest.builder()
            if space_id:
                builder.space_id(space_id)

            request = builder \
                .query(query) \
                .build()

            response = self._client.wiki.v2.node.search(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            results = response.data.get('results', []) if response.data else []
            if not results:
                return f"No results found for '{query}'"

            output = [f"Search results for '{query}':"]
            for i, item in enumerate(results[:limit]):
                title = item.get('title', 'Untitled')
                obj_token = item.get('obj_token', '')
                if obj_token:
                    url = f"https://feishu.cn/wiki/{obj_token}"
                    output.append(f"{i+1}. {title}\n   URL: {url}")
                else:
                    output.append(f"{i+1}. {title}")

            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error searching wiki: {e}")
            return f"Error: {str(e)}"
