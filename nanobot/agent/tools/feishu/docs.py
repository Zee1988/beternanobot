"""Feishu document tools: read, create, append."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

try:
    from lark_oapi import docx
    from lark_oapi.api.docx.v1 import (
        AppendBlockRequest,
        AppendBlockRequestBody,
        CreateDocumentRequest,
        CreateDocumentRequestBody,
        GetDocumentRequest,
    )
    FEISHU_DOCX_AVAILABLE = True
except ImportError:
    FEISHU_DOCX_AVAILABLE = False
    docx = None


class FeishuDocRead(Tool):
    """Tool to read Feishu document content."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_doc_read"

    @property
    def description(self) -> str:
        return "Read the content of a Feishu document by its token. Returns the document text content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "document_token": {
                    "type": "string",
                    "description": "The document token (starts with '-') from the document URL"
                }
            },
            "required": ["document_token"]
        }

    async def execute(self, document_token: str, **kwargs: Any) -> str:
        if not FEISHU_DOCX_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            # Add '-' prefix if not present
            if not document_token.startswith("-"):
                document_token = "-" + document_token

            request = GetDocumentRequest.builder() \
                .document_id(document_token) \
                .build()

            response = self._client.docx.v1.document.get(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            # Extract text content from blocks
            data = response.data
            if hasattr(data, 'document') and hasattr(data.document, 'body'):
                blocks = data.document.body.get('blocks', [])
                return self._extract_text_from_blocks(blocks)

            return "Document found but content unavailable"
        except Exception as e:
            logger.error(f"Error reading Feishu doc: {e}")
            return f"Error reading document: {str(e)}"

    def _extract_text_from_blocks(self, blocks: list) -> str:
        """Extract text content from document blocks."""
        texts = []
        for block in blocks:
            block_type = block.get('type', '')
            if block_type == 'text':
                text_content = block.get('text', {}).get('content', '')
                if text_content:
                    texts.append(text_content)
            elif block_type == 'paragraph':
                elements = block.get('paragraph', {}).get('elements', [])
                for elem in elements:
                    if 'text' in elem:
                        text_content = elem['text'].get('content', '')
                        if text_content:
                            texts.append(text_content)
        return "\n".join(texts) if texts else "(empty document)"


class FeishuDocCreate(Tool):
    """Tool to create a new Feishu document."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_doc_create"

    @property
    def description(self) -> str:
        return "Create a new Feishu document. Returns the new document URL."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Document title"
                },
                "content": {
                    "type": "string",
                    "description": "Initial document content (optional)"
                }
            },
            "required": ["title"]
        }

    async def execute(self, title: str, content: str = "", **kwargs: Any) -> str:
        if not FEISHU_DOCX_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            request = CreateDocumentRequest.builder() \
                .request_body(
                    CreateDocumentRequestBody.builder()
                    .title(title)
                    .build()
                ).build()

            response = self._client.docx.v1.document.create(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            doc_token = response.data.document.token
            url = f"https://feishu.cn/document/{doc_token}"

            # If content provided, append it
            if content:
                await self._append_content(doc_token, content)
                url += "\n(Content appended)"

            return f"Document created: {url}"
        except Exception as e:
            logger.error(f"Error creating Feishu doc: {e}")
            return f"Error creating document: {str(e)}"

    async def _append_content(self, document_token: str, content: str) -> None:
        """Append content to a document."""
        if not self._client:
            return

        try:
            request = AppendBlockRequest.builder() \
                .document_id(document_token) \
                .request_body(
                    AppendBlockRequestBody.builder()
                    .block(
                        {
                            "type": "text",
                            "text": {"content": content}
                        }
                    )
                    .build()
                ).build()

            self._client.docx.v1.block.append(request)
        except Exception as e:
            logger.warning(f"Failed to append content: {e}")


class FeishuDocAppend(Tool):
    """Tool to append content to an existing Feishu document."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_doc_append"

    @property
    def description(self) -> str:
        return "Append text content to an existing Feishu document."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "document_token": {
                    "type": "string",
                    "description": "The document token (starts with '-') from the document URL"
                },
                "content": {
                    "type": "string",
                    "description": "Text content to append"
                }
            },
            "required": ["document_token", "content"]
        }

    async def execute(self, document_token: str, content: str, **kwargs: Any) -> str:
        if not FEISHU_DOCX_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            if not document_token.startswith("-"):
                document_token = "-" + document_token

            request = AppendBlockRequest.builder() \
                .document_id(document_token) \
                .request_body(
                    AppendBlockRequestBody.builder()
                    .block(
                        {
                            "type": "text",
                            "text": {"content": content}
                        }
                    )
                    .build()
                ).build()

            response = self._client.docx.v1.block.append(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            return f"Content appended to document {document_token}"
        except Exception as e:
            logger.error(f"Error appending to Feishu doc: {e}")
            return f"Error appending content: {str(e)}"
