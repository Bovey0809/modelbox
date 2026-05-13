#!/usr/bin/env python3
"""End-to-end smoke test for the tennis_action demo. Runs the configured
graph and asserts the result files exist + are structurally valid.

Usage (typical):
    python3 test_tennis_action.py \\
        --modelbox-tool build/release/usr/local/bin/modelbox-tool \\
        --graph build/src/demo/tennis_action/graph/tennis_action_cpu_stub.toml

Returns 0 on success, non-zero on first failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 600) -> int:
    print(f"[run] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True,
                         text=True)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0 and res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelbox-tool", required=True,
                   help="Path to the built modelbox-tool binary.")
    p.add_argument("--graph", required=True,
                   help="Path to the configured graph .toml.")
    p.add_argument("--out-json", default="/tmp/tennis_actions.json")
    p.add_argument("--out-overlay", default="/tmp/tennis_actions_overlay.mp4")
    p.add_argument("--out-dropped",
                   default="/tmp/tennis_actions_dropped.json")
    args = p.parse_args()

    # Clean previous run.
    for f in (args.out_json, args.out_overlay, args.out_dropped):
        Path(f).unlink(missing_ok=True)

    # Verify prereqs.
    tool = Path(args.modelbox_tool)
    graph = Path(args.graph)
    if not tool.exists():
        print(f"[FAIL] modelbox-tool not found at {tool}", file=sys.stderr)
        return 10
    if not graph.exists():
        print(f"[FAIL] graph not found at {graph}", file=sys.stderr)
        return 11

    rc = _run([str(tool), "flow", "run", "-name", "TennisAction",
               "-path", str(graph)])
    if rc != 0:
        print(f"[FAIL] modelbox-tool exited {rc}", file=sys.stderr)
        return 1

    # 1. JSON exists and parses.
    j_path = Path(args.out_json)
    if not j_path.exists():
        print(f"[FAIL] {j_path} not produced", file=sys.stderr)
        return 2
    body = json.loads(j_path.read_text())
    if "summary" not in body or "hits" not in body:
        print("[FAIL] JSON missing summary/hits keys", file=sys.stderr)
        return 3
    print(f"[ok] {len(body['hits'])} confirmed hits; "
          f"{body['summary']['dropped']} drops")

    # 2. Overlay decodable (best-effort — skipped if not produced).
    o_path = Path(args.out_overlay)
    if o_path.exists():
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,duration", "-of",
             "default=noprint_wrappers=1", str(o_path)],
            capture_output=True, text=True
        )
        if res.returncode != 0 or "codec_type=video" not in res.stdout:
            print(f"[FAIL] overlay not decodable: {res.stderr}",
                  file=sys.stderr)
            return 5
        print(f"[ok] {o_path.stat().st_size} bytes overlay")
    else:
        print(f"[skip] {o_path} not produced "
              "(expected if source_video=' ' or cv2 missing)")

    # 3. Dropped JSON exists.
    d_path = Path(args.out_dropped)
    if not d_path.exists():
        print(f"[FAIL] {d_path} not produced", file=sys.stderr)
        return 6
    json.loads(d_path.read_text())  # parses
    print("[ok] dropped JSON valid")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
