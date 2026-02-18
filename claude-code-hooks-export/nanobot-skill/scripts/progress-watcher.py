#!/usr/bin/env python3
"""Progress watcher for Claude Code tasks.

Monitors task-output.txt and sends periodic progress updates
directly to Telegram/Feishu via their APIs. No agent involvement.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

NANOBOT_CONFIG = os.path.expanduser("~/.nanobot/config.json")
POLL_INTERVAL = 30  # seconds between checks
REPORT_INTERVAL = 60  # minimum seconds between progress reports
MAX_WATCH_TIME = 660  # stop watching after 11 minutes (task timeout is 10min)
CHUNK_SIZE = 500  # max chars per progress update


def load_nanobot_config():
    try:
        with open(NANOBOT_CONFIG) as f:
            return json.load(f)
    except Exception:
        return {}


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"telegram send failed: {e}", file=sys.stderr)
def get_feishu_token(app_id, app_secret):
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read()).get("tenant_access_token", "")
    except Exception as e:
        print(f"feishu token failed: {e}", file=sys.stderr)
        return ""


def send_feishu(app_id, app_secret, chat_id, text):
    token = get_feishu_token(app_id, app_secret)
    if not token:
        return
    # Determine receive_id_type
    if chat_id.startswith("oc_"):
        id_type = "chat_id"
    else:
        id_type = "open_id"
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={id_type}"
    content = json.dumps({"text": text})
    data = json.dumps({
        "receive_id": chat_id,
        "msg_type": "text",
        "content": content,
    }).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    })
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"feishu send failed: {e}", file=sys.stderr)
def send_message(channel, chat_id, text, config):
    channels = config.get("channels", {})
    if channel == "telegram":
        token = channels.get("telegram", {}).get("token", "")
        if token:
            send_telegram(token, chat_id, text)
    elif channel == "feishu":
        feishu_cfg = channels.get("feishu", {})
        app_id = feishu_cfg.get("appId", "")
        app_secret = feishu_cfg.get("appSecret", "")
        if app_id and app_secret:
            send_feishu(app_id, app_secret, chat_id, text)


def read_output(task_dir):
    output_file = os.path.join(task_dir, "task-output.txt")
    try:
        with open(output_file) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def read_meta(task_dir):
    meta_file = os.path.join(task_dir, "task-meta.json")
    try:
        with open(meta_file) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    if len(sys.argv) < 4:
        print("Usage: progress-watcher.py <task_dir> <channel> <chat_id>", file=sys.stderr)
        sys.exit(1)

    task_dir = sys.argv[1]
    channel = sys.argv[2]
    chat_id = sys.argv[3]

    config = load_nanobot_config()
    meta = read_meta(task_dir)
    task_name = meta.get("name", "") or meta.get("task_id", "unknown")

    last_len = 0
    last_report_time = 0
    start_time = time.time()

    # Initial notification
    send_message(channel, chat_id, f"⏳ Task [{task_name}] started...", config)

    while time.time() - start_time < MAX_WATCH_TIME:
        time.sleep(POLL_INTERVAL)

        # Check if task completed
        meta = read_meta(task_dir)
        status = meta.get("status", "running")

        output = read_output(task_dir)
        current_len = len(output)

        if status in ("completed", "failed"):
            emoji = "✅" if status == "completed" else "❌"
            snippet = output[-CHUNK_SIZE:].strip() if output else "(no output)"
            send_message(channel, chat_id,
                f"{emoji} Task [{task_name}] {status}.\n\n{snippet}", config)
            return

        # Send progress if new content and enough time passed
        now = time.time()
        if current_len > last_len and (now - last_report_time) >= REPORT_INTERVAL:
            new_content = output[last_len:][-CHUNK_SIZE:].strip()
            elapsed = int(now - start_time)
            send_message(channel, chat_id,
                f"🔄 Task [{task_name}] in progress ({elapsed}s)...\n\n{new_content}", config)
            last_len = current_len
            last_report_time = now

    # Timeout
    send_message(channel, chat_id,
        f"⏰ Task [{task_name}] watcher timed out after {MAX_WATCH_TIME}s.", config)


if __name__ == "__main__":
    main()

