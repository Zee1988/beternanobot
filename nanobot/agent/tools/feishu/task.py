"""Feishu Task tools."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

try:
    from lark_oapi import task
    from lark_oapi.api.task.v1 import (
        CreateTaskRequest,
        CreateTaskRequestBody,
        ListTaskRequest,
        ListTaskRequestBody,
        UpdateTaskRequest,
        UpdateTaskRequestBody,
    )
    FEISHU_TASK_AVAILABLE = True
except ImportError:
    FEISHU_TASK_AVAILABLE = False
    task = None


class FeishuTaskCreate(Tool):
    """Tool to create a new Feishu Task."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_task_create"

    @property
    def description(self) -> str:
        return "Create a new task in Feishu Tasks. Requires tasklist_id and task title."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasklist_id": {
                    "type": "string",
                    "description": "The task list ID"
                },
                "title": {
                    "type": "string",
                    "description": "Task title"
                },
                "description": {
                    "type": "string",
                    "description": "Task description (optional)"
                },
                "due": {
                    "type": "string",
                    "description": "Due date in ISO format, e.g. '2024-12-31' (optional)"
                },
                "start": {
                    "type": "string",
                    "description": "Start date in ISO format (optional)"
                }
            },
            "required": ["tasklist_id", "title"]
        }

    async def execute(self, tasklist_id: str, title: str, description: str = "", due: str = "", start: str = "", **kwargs: Any) -> str:
        if not FEISHU_TASK_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            builder = CreateTaskRequest.builder() \
                .request_body(
                    CreateTaskRequestBody.builder()
                    .tasklist_id(tasklist_id)
                    .title(title)
                    .build()
                )

            if description:
                builder.request_body.description(description)
            if due:
                builder.request_body.due(due)
            if start:
                builder.request_body.start(start)

            request = builder.build()

            response = self._client.task.v1.task.create(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            task_id = response.data.task.id if response.data and response.data.task else "unknown"
            return f"Task created: {title} (ID: {task_id})"
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return f"Error: {str(e)}"


class FeishuTaskList(Tool):
    """Tool to list tasks from a Feishu task list."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_task_list"

    @property
    def description(self) -> str:
        return "List tasks from a Feishu task list."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasklist_id": {
                    "type": "string",
                    "description": "The task list ID (optional, lists all if not provided)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum tasks to return (default 20)",
                    "default": 20
                },
                "completed": {
                    "type": "boolean",
                    "description": "Show only completed tasks (default false)"
                }
            },
            "required": []
        }

    async def execute(self, tasklist_id: str = "", limit: int = 20, completed: bool = False, **kwargs: Any) -> str:
        if not FEISHU_TASK_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            builder = ListTaskRequest.builder() \
                .request_body(
                    ListTaskRequestBody.builder()
                    .limit(limit)
                    .build()
                )

            if tasklist_id:
                builder.request_body.tasklist_id(tasklist_id)
            if completed:
                builder.request_body.completed(True)

            request = builder.build()

            response = self._client.task.v1.task.list(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            tasks = response.data.get('items', []) if response.data else []
            if not tasks:
                return "No tasks found"

            results = [f"Tasks ({len(tasks)}):"]
            for t in tasks:
                task_id = t.get('id', 'unknown')
                title = t.get('title', 'Untitled')
                status = "✓" if t.get('completed') else "○"
                due = t.get('due', '')
                due_str = f" (due: {due})" if due else ""
                results.append(f"{status} {title}{due_str} [ID:{task_id}]")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return f"Error: {str(e)}"


class FeishuTaskComplete(Tool):
    """Tool to mark a Feishu task as complete."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_task_complete"

    @property
    def description(self) -> str:
        return "Mark a Feishu task as completed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to complete"
                }
            },
            "required": ["task_id"]
        }

    async def execute(self, task_id: str, **kwargs: Any) -> str:
        if not FEISHU_TASK_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            request = UpdateTaskRequest.builder() \
                .task_id(task_id) \
                .request_body(
                    UpdateTaskRequestBody.builder()
                    .completed(True)
                    .build()
                ).build()

            response = self._client.task.v1.task.update(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            return f"Task {task_id} marked as completed"
        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return f"Error: {str(e)}"
