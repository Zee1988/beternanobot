"""Feishu tools module."""

from nanobot.agent.tools.feishu.bitable import (
    FeishuBitableInsert,
    FeishuBitableQuery,
    FeishuBitableSearch,
    FeishuBitableUpdate,
)
from nanobot.agent.tools.feishu.docs import FeishuDocAppend, FeishuDocCreate, FeishuDocRead
from nanobot.agent.tools.feishu.drive import (
    FeishuDriveCreateFolder,
    FeishuDriveList,
    FeishuDriveUpload,
)
from nanobot.agent.tools.feishu.task import FeishuTaskComplete, FeishuTaskCreate, FeishuTaskList
from nanobot.agent.tools.feishu.wiki import FeishuWikiListSpaces, FeishuWikiSearch

__all__ = [
    # Docs
    "FeishuDocRead",
    "FeishuDocCreate",
    "FeishuDocAppend",
    # Wiki
    "FeishuWikiListSpaces",
    "FeishuWikiSearch",
    # Drive
    "FeishuDriveList",
    "FeishuDriveUpload",
    "FeishuDriveCreateFolder",
    # Bitable
    "FeishuBitableQuery",
    "FeishuBitableInsert",
    "FeishuBitableUpdate",
    "FeishuBitableSearch",
    # Task
    "FeishuTaskCreate",
    "FeishuTaskList",
    "FeishuTaskComplete",
]
