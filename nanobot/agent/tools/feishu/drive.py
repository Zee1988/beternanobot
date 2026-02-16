"""Feishu Cloud Drive tools."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

try:
    from lark_oapi import drive
    from lark_oapi.api.drive.v1 import (
        CreateFolderFileRequest,
        CreateFolderFileRequestBody,
        ListFileRequest,
        ListFileRequestBody,
        UploadAllFileRequest,
        UploadAllFileRequestBody,
    )
    FEISHU_DRIVE_AVAILABLE = True
except ImportError:
    FEISHU_DRIVE_AVAILABLE = False
    drive = None


class FeishuDriveList(Tool):
    """Tool to list files in Feishu Cloud Drive."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_drive_list"

    @property
    def description(self) -> str:
        return "List files in Feishu Cloud Drive root or a specific folder."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "folder_token": {
                    "type": "string",
                    "description": "Folder token to list (optional, defaults to root)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to return (default 50)",
                    "default": 50
                }
            },
            "required": []
        }

    async def execute(self, folder_token: str = "", limit: int = 50, **kwargs: Any) -> str:
        if not FEISHU_DRIVE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            builder = ListFileRequest.builder()
            if folder_token:
                builder.folder_token(folder_token)
            else:
                builder.folder_token("-")  # Root folder

            request = builder \
                .request_body(
                    ListFileRequestBody.builder()
                    .limit(limit)
                    .build()
                ).build()

            response = self._client.drive.v1.file.list(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            files = response.data.get('files', []) if response.data else []
            if not files:
                return "No files found in this folder"

            results = ["Files:"]
            for f in files:
                name = f.get('name', 'Unnamed')
                ftype = f.get('type', 'unknown')
                if ftype == 'folder':
                    results.append(f"📁 {name}/")
                else:
                    results.append(f"📄 {name}")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error listing drive files: {e}")
            return f"Error: {str(e)}"


class FeishuDriveUpload(Tool):
    """Tool to upload a file to Feishu Cloud Drive."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_drive_upload"

    @property
    def description(self) -> str:
        return "Upload a file to Feishu Cloud Drive. Provide file content as text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Name of the file to create"
                },
                "content": {
                    "type": "string",
                    "description": "Content of the file"
                },
                "folder_token": {
                    "type": "string",
                    "description": "Target folder token (optional, defaults to root)"
                },
                "file_type": {
                    "type": "string",
                    "description": "File type: txt, sheet, doc, etc. (default: txt)"
                }
            },
            "required": ["file_name", "content"]
        }

    async def execute(self, file_name: str, content: str, folder_token: str = "", file_type: str = "txt", **kwargs: Any) -> str:
        if not FEISHU_DRIVE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            request = UploadAllFileRequest.builder() \
                .request_body(
                    UploadAllFileRequestBody.builder()
                    .file_name(file_name)
                    .file_type(file_type)
                    .content(content.encode('utf-8'))
                    .folder_token(folder_token if folder_token else None)
                    .build()
                ).build()

            response = self._client.drive.v1.file.upload_all(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            file_token = response.data.file.token
            return f"File uploaded successfully: {file_name} (token: {file_token})"
        except Exception as e:
            logger.error(f"Error uploading to drive: {e}")
            return f"Error: {str(e)}"


class FeishuDriveCreateFolder(Tool):
    """Tool to create a folder in Feishu Cloud Drive."""

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def name(self) -> str:
        return "feishu_drive_create_folder"

    @property
    def description(self) -> str:
        return "Create a new folder in Feishu Cloud Drive."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "folder_name": {
                    "type": "string",
                    "description": "Name of the folder to create"
                },
                "parent_folder_token": {
                    "type": "string",
                    "description": "Parent folder token (optional, defaults to root)"
                }
            },
            "required": ["folder_name"]
        }

    async def execute(self, folder_name: str, parent_folder_token: str = "", **kwargs: Any) -> str:
        if not FEISHU_DRIVE_AVAILABLE or not self._client:
            return "Error: lark_oapi SDK not available or client not initialized"

        try:
            request = CreateFolderFileRequest.builder() \
                .request_body(
                    CreateFolderFileRequestBody.builder()
                    .name(folder_name)
                    .folder_token(parent_folder_token if parent_folder_token else None)
                    .build()
                ).build()

            response = self._client.drive.v1.file.create_folder(request)

            if not response.success():
                return f"Error: code={response.code}, msg={response.msg}"

            folder_token = response.data.file.token
            return f"Folder created: {folder_name} (token: {folder_token})"
        except Exception as e:
            logger.error(f"Error creating folder: {e}")
            return f"Error: {str(e)}"
