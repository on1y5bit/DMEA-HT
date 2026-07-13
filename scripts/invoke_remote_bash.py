#!/usr/bin/env python3
"""Run a UTF-8 Bash script over SSH without PowerShell interpolation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--host", default="10.21.71.74")
    parser.add_argument("--user", default="linruixin")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--askpass", type=Path)
    parser.add_argument("--known-hosts", type=Path, default=Path(".tmp_known_hosts"))
    parser.add_argument("--connect-timeout", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ssh_executable = shutil.which("ssh")
    if ssh_executable is None:
        raise SystemExit("ssh executable was not found on PATH")

    script_path = args.script.resolve(strict=True)
    script_bytes = script_path.read_bytes()
    try:
        script_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"remote script is not valid UTF-8: {script_path}: {exc}") from exc

    environment = os.environ.copy()
    if args.askpass is not None:
        environment["SSH_ASKPASS"] = str(args.askpass.resolve(strict=True))
        environment["SSH_ASKPASS_REQUIRE"] = "force"
        environment["DISPLAY"] = "codex"

    command = [
        ssh_executable,
        "-T",
        "-p",
        str(args.port),
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"UserKnownHostsFile={args.known_hosts.resolve()}",
        f"{args.user}@{args.host}",
        "bash",
        "-se",
    ]
    completed = subprocess.run(
        command,
        input=script_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

