"""Lightweight notification sender for watchdog alerts.

Sends messages directly via platform HTTP APIs — does **not** depend on the
channel stack being alive.  Provides both async and sync entry-points so
callers inside a dead event-loop can still deliver notifications.
"""

from __future__ import annotations

import json as _json
import smtplib
from email.mime.text import MIMEText
from typing import Any

import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# Per-platform senders (async)
# ---------------------------------------------------------------------------

async def _send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json={"chat_id": chat_id, "text": text})
        r.raise_for_status()


async def _send_discord(token: str, user_id: str, text: str) -> None:
    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=15) as client:
        # Open / retrieve DM channel
        r = await client.post(
            "https://discord.com/api/v10/users/@me/channels",
            headers=headers,
            json={"recipient_id": user_id},
        )
        r.raise_for_status()
        dm_channel_id = r.json()["id"]
        # Send message
        r = await client.post(
            f"https://discord.com/api/v10/channels/{dm_channel_id}/messages",
            headers=headers,
            json={"content": text},
        )
        r.raise_for_status()


async def _send_feishu(app_id: str, app_secret: str, user_id: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        # Get tenant access token
        r = await client.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret},
        )
        r.raise_for_status()
        token = r.json().get("tenant_access_token", "")
        # Send message
        r = await client.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
            headers={"Authorization": f"Bearer {token}"},
            json={"receive_id": user_id, "msg_type": "text", "content": _json.dumps({"text": text})},
        )
        r.raise_for_status()


async def _send_dingtalk(client_id: str, client_secret: str, staff_id: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.dingtalk.com/v1.0/oauth2/accessToken",
            json={"appKey": client_id, "appSecret": client_secret},
        )
        r.raise_for_status()
        token = r.json().get("accessToken", "")
        r = await client.post(
            "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend",
            headers={"x-acs-dingtalk-access-token": token},
            json={"robotCode": client_id, "userIds": [staff_id], "msgKey": "sampleText",
                   "msgParam": _json.dumps({"content": text})},
        )
        r.raise_for_status()


async def _send_slack(bot_token: str, user_id: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {bot_token}"},
            json={"channel": user_id, "text": text},
        )
        r.raise_for_status()


def _send_email_sync(cfg: Any, to_addr: str, text: str) -> None:
    """Synchronous email send via SMTP — usable outside an event-loop."""
    msg = MIMEText(text)
    msg["Subject"] = "[nanobot] Watchdog Alert"
    msg["From"] = cfg.from_address or cfg.smtp_username
    msg["To"] = to_addr
    if cfg.smtp_use_ssl:
        server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=15)
    else:
        server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15)
        if cfg.smtp_use_tls:
            server.starttls()
    try:
        server.login(cfg.smtp_username, cfg.smtp_password)
        server.send_message(msg)
    finally:
        server.quit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def notify_all(config: Any, text: str) -> None:
    """Send *text* to the first ``allow_from`` user of every enabled channel."""
    ch = config.channels

    if ch.telegram.enabled and ch.telegram.token and ch.telegram.allow_from:
        try:
            await _send_telegram(ch.telegram.token, ch.telegram.allow_from[0], text)
            logger.info("Watchdog notification sent via Telegram")
        except Exception as e:
            logger.warning(f"Telegram notify failed: {e}")

    if ch.discord.enabled and ch.discord.token and ch.discord.allow_from:
        try:
            await _send_discord(ch.discord.token, ch.discord.allow_from[0], text)
            logger.info("Watchdog notification sent via Discord")
        except Exception as e:
            logger.warning(f"Discord notify failed: {e}")

    if ch.feishu.enabled and ch.feishu.app_id and ch.feishu.allow_from:
        try:
            await _send_feishu(
                ch.feishu.app_id, ch.feishu.app_secret,
                ch.feishu.allow_from[0], text,
            )
            logger.info("Watchdog notification sent via Feishu")
        except Exception as e:
            logger.warning(f"Feishu notify failed: {e}")

    if ch.dingtalk.enabled and ch.dingtalk.client_id and ch.dingtalk.allow_from:
        try:
            await _send_dingtalk(
                ch.dingtalk.client_id, ch.dingtalk.client_secret,
                ch.dingtalk.allow_from[0], text,
            )
            logger.info("Watchdog notification sent via DingTalk")
        except Exception as e:
            logger.warning(f"DingTalk notify failed: {e}")

    if ch.slack.enabled and ch.slack.bot_token and ch.slack.dm.allow_from:
        try:
            await _send_slack(ch.slack.bot_token, ch.slack.dm.allow_from[0], text)
            logger.info("Watchdog notification sent via Slack")
        except Exception as e:
            logger.warning(f"Slack notify failed: {e}")

    if ch.email.enabled and ch.email.smtp_host and ch.email.allow_from:
        try:
            _send_email_sync(ch.email, ch.email.allow_from[0], text)
            logger.info("Watchdog notification sent via Email")
        except Exception as e:
            logger.warning(f"Email notify failed: {e}")


def notify_all_sync(config: Any, text: str) -> None:
    """Synchronous wrapper — creates a temporary event-loop if needed."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, notify_all(config, text)).result(timeout=30)
    else:
        asyncio.run(notify_all(config, text))