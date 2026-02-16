"""Feishu Bitable (multi-dimensional table) tools."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

try:
    from lark_oapi import bitable
    from lark_oapi.api.bitable.v1 import (
        CreateAppTableRecordRequest,
        CreateAppTableRecordRequestBody,
        ListAppTableRecordRequest,
        ListAppTableRecordRequestBody,
        SearchAppTableRecordRequest,
        SearchAppTableRecordRequestBody,
        UpdateAppTableRecordRequest,
        UpdateAppTableRecordRequestBody,
    )
    FEISHU_BITABLE_AVAILABLE = True
except ImportError:
    FEISHU_BITABLE_AVAILABLE = False
    bitable = None


class FeishuBitableQuery(Tool):
    """Tool to query records from a Feishu Bitable."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_bitable_query"

    @property
    def description(self) -> str:
        return "Query records from a Feishu Bitable (multi-dimensional table). Requires app_token and table_id."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "The bitable app token (starts with '-')"
                },
                "table_id": {
                    "type": "string",
                    "description": "The table ID within the bitable"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum records to return (default 50)",
                    "default": 50
                },
                "filter": {
                    "type": "string",
                    "description": "Filter formula (optional, e.g. 'Name == \"John\"')"
                }
            },
            "required": ["app_token", "table_id"]
        }

    async def execute(self, app_token: str, table_id: str, limit: int = 50, filter: str = "", **kwargs: Any) -> str:
        if not FEISHU_BITABLE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            # Ensure token format
            if not app_token.startswith("-"):
                app_token = "-" + app_token

            builder = ListAppTableRecordRequest.builder() \
                .app_token(app_token) \
                .table_id(table_id) \
                .request_body(
                    ListAppTableRecordRequestBody.builder()
                    .limit(limit)
                    .build()
                )

            request = builder.build()

            response = self._client.bitable.v1.app_table_record.list(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            records = response.data.get('items', []) if response.data else []
            if not records:
                return "No records found"

            results = [f"Records (showing {len(records)} of {response.data.get('total', '?')}):"]
            for i, record in enumerate(records):
                record_id = record.get('record_id', 'unknown')
                fields = record.get('fields', {})
                # Show first few fields as preview
                field_preview = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:3])
                results.append(f"{i+1}. [ID:{record_id}] {field_preview}")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error querying bitable: {e}")
            return f"Error: {str(e)}"


class FeishuBitableInsert(Tool):
    """Tool to insert a record into a Feishu Bitable."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_bitable_insert"

    @property
    def description(self) -> str:
        return "Insert a new record into a Feishu Bitable. Provide field values as a JSON object."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "The bitable app token (starts with '-')"
                },
                "table_id": {
                    "type": "string",
                    "description": "The table ID within the bitable"
                },
                "fields": {
                    "type": "string",
                    "description": "JSON object of field names and values, e.g. '{\"Name\": \"John\", \"Age\": 30}'"
                }
            },
            "required": ["app_token", "table_id", "fields"]
        }

    async def execute(self, app_token: str, table_id: str, fields: str, **kwargs: Any) -> str:
        if not FEISHU_BITABLE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            import json

            if not app_token.startswith("-"):
                app_token = "-" + app_token

            # Parse fields from JSON string
            fields_dict = json.loads(fields)

            request = CreateAppTableRecordRequest.builder() \
                .app_token(app_token) \
                .table_id(table_id) \
                .request_body(
                    CreateAppTableRecordRequestBody.builder()
                    .records([{"fields": fields_dict}])
                    .build()
                ).build()

            response = self._client.bitable.v1.app_table_record.create(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            record_id = response.data.records[0].record_id if response.data and response.data.records else "unknown"
            return f"Record created successfully (ID: {record_id})"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in fields: {str(e)}"
        except Exception as e:
            logger.error(f"Error inserting into bitable: {e}")
            return f"Error: {str(e)}"


class FeishuBitableUpdate(Tool):
    """Tool to update a record in a Feishu Bitable."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_bitable_update"

    @property
    def description(self) -> str:
        return "Update an existing record in a Feishu Bitable. Provide record_id and field values as JSON."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "The bitable app token (starts with '-')"
                },
                "table_id": {
                    "type": "string",
                    "description": "The table ID within the bitable"
                },
                "record_id": {
                    "type": "string",
                    "description": "The record ID to update"
                },
                "fields": {
                    "type": "string",
                    "description": "JSON object of field names and new values"
                }
            },
            "required": ["app_token", "table_id", "record_id", "fields"]
        }

    async def execute(self, app_token: str, table_id: str, record_id: str, fields: str, **kwargs: Any) -> str:
        if not FEISHU_BITABLE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            import json

            if not app_token.startswith("-"):
                app_token = "-" + app_token

            fields_dict = json.loads(fields)

            request = UpdateAppTableRecordRequest.builder() \
                .app_token(app_token) \
                .table_id(table_id) \
                .record_id(record_id) \
                .request_body(
                    UpdateAppTableRecordRequestBody.builder()
                    .records([{"fields": fields_dict}])
                    .build()
                ).build()

            response = self._client.bitable.v1.app_table_record.update(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            return f"Record {record_id} updated successfully"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in fields: {str(e)}"
        except Exception as e:
            logger.error(f"Error updating bitable: {e}")
            return f"Error: {str(e)}"


class FeishuBitableSearch(Tool):
    """Tool to search records in a Feishu Bitable."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_bitable_search"

    @property
    def description(self) -> str:
        return "Search records in a Feishu Bitable using a filter formula."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "The bitable app token (starts with '-')"
                },
                "table_id": {
                    "type": "string",
                    "description": "The table ID within the bitable"
                },
                "filter": {
                    "type": "string",
                    "description": "Filter formula, e.g. 'Name == \"John\"' or 'Age > 25'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum records to return (default 50)",
                    "default": 50
                }
            },
            "required": ["app_token", "table_id", "filter"]
        }

    async def execute(self, app_token: str, table_id: str, filter: str, limit: int = 50, **kwargs: Any) -> str:
        if not FEISHU_BITABLE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            if not app_token.startswith("-"):
                app_token = "-" + app_token

            request = SearchAppTableRecordRequest.builder() \
                .app_token(app_token) \
                .table_id(table_id) \
                .request_body(
                    SearchAppTableRecordRequestBody.builder()
                    .filter(filter)
                    .limit(limit)
                    .build()
                ).build()

            response = self._client.bitable.v1.app_table_record.search(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            records = response.data.get('items', []) if response.data else []
            if not records:
                return "No matching records found"

            results = [f"Found {len(records)} records:"]
            for i, record in enumerate(records):
                record_id = record.get('record_id', 'unknown')
                fields = record.get('fields', {})
                field_preview = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:3])
                results.append(f"{i+1}. [ID:{record_id}] {field_preview}")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error searching bitable: {e}")
            return f"Error: {str(e)}"
