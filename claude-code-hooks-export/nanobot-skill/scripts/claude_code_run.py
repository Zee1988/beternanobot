#!/usr/bin/env python3
"""PTY runner for Claude Code on macOS.

Adapted from win4r/claude-code-hooks for macOS compatibility.
Supports headless (-p) mode and tmux session mode.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def strip_ansi(text):
    """Remove ANSI escape sequences and control characters from PTY output."""
    # Remove ANSI escape sequences
    text = re.sub(r'\x1b\[[^@-~]*[@-~]', '', text)
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)  # OSC sequences
    text = re.sub(r'\x1b[()][AB012]', '', text)  # charset sequences
    text = re.sub(r'\x1b\[[\?]?[0-9;]*[a-zA-Z]', '', text)
    # Remove control chars except newline/tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Remove ^D artifacts
    text = re.sub(r'\^D', '', text)
    return text.strip()


def find_claude_binary():
    """Locate the claude CLI binary."""
    explicit = os.environ.get("CLAUDE_BINARY")
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    candidates = [
        os.path.expanduser("~/.local/bin/claude"),
        os.path.expanduser("~/.npm-global/bin/claude"),
        "/opt/homebrew/bin/claude",
        "/usr/local/bin/claude",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    print("ERROR: claude binary not found", file=sys.stderr)
    sys.exit(1)


def build_command(args, claude_bin):
    """Build the claude CLI command from arguments."""
    cmd = [claude_bin, "-p", args.prompt, "--output-format", "text"]
    if args.permission_mode:
        cmd.extend(["--permission-mode", args.permission_mode])
    if args.allowed_tools:
        for tool in args.allowed_tools:
            cmd.extend(["--allowedTools", tool])
    if args.model:
        cmd.extend(["--model", args.model])
    return cmd


def build_env(args):
    """Build environment variables for the subprocess."""
    env = os.environ.copy()
    # Remove nested session detection so Claude Code can be launched from any context
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    if args.agent_teams:
        env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
    if args.teammate_mode:
        env["CLAUDE_CODE_TEAMMATE_MODE"] = args.teammate_mode
    return env


def run_headless(args, claude_bin):
    """Run claude in headless mode using macOS script(1) for PTY."""
    cmd = build_command(args, claude_bin)
    env = build_env(args)
    workdir = args.workdir or os.getcwd()

    # macOS script syntax: script -q /dev/null <cmd>
    # This provides a PTY wrapper so claude thinks it has a terminal
    script_cmd = ["script", "-q", "/dev/null"] + cmd

    result = subprocess.run(
        script_cmd,
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout or 600,
    )
    return result


def run_tmux(args, claude_bin):
    """Run claude inside a tmux session."""
    if not shutil.which("tmux"):
        print("ERROR: tmux not found", file=sys.stderr)
        sys.exit(1)

    session_name = args.session_name or f"claude-{os.getpid()}"
    cmd = build_command(args, claude_bin)
    env = build_env(args)
    workdir = args.workdir or os.getcwd()

    # Create tmux session
    full_cmd = " ".join(cmd)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-c", workdir, full_cmd],
        env=env,
        check=True,
    )
    print(json.dumps({"tmux_session": session_name, "status": "started"}))
    return None


def main():
    parser = argparse.ArgumentParser(description="Claude Code PTY Runner for macOS")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt to send to Claude Code")
    parser.add_argument("-w", "--workdir", help="Working directory")
    parser.add_argument("--permission-mode", help="Permission mode (plan, auto-edit, full-auto)")
    parser.add_argument("--allowed-tools", nargs="*", help="Allowed tools list")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--agent-teams", action="store_true", help="Enable Agent Teams")
    parser.add_argument("--teammate-mode", help="Teammate mode value")
    parser.add_argument("--session-name", help="tmux session name (tmux mode only)")
    parser.add_argument("--tmux", action="store_true", help="Run in tmux session")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    parser.add_argument("--output-file", help="Write output to file instead of stdout")
    args = parser.parse_args()

    claude_bin = find_claude_binary()

    if args.tmux:
        run_tmux(args, claude_bin)
        return

    try:
        result = run_headless(args, claude_bin)
    except subprocess.TimeoutExpired:
        print(json.dumps({"error": "timeout", "timeout_seconds": args.timeout}), file=sys.stderr)
        sys.exit(124)

    output = result.stdout or ""
    output = strip_ansi(output)
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
    else:
        print(output)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
