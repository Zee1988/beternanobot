"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
                "plan_file": {
                    "type": "string",
                    "description": (
                        "Optional path to a plan file (markdown, JSON, etc.) "
                        "that guides the subagent's execution step-by-step"
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        plan_file: str | None = None,
        *,
        _origin_channel: str = "cli",
        _origin_chat_id: str = "direct",
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent. Origin context injected by caller, not stored as state."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=_origin_channel,
            origin_chat_id=_origin_chat_id,
            plan_file=plan_file,
        )
