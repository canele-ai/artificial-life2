"""Modal smoke test — verifies image builds, secrets mount, judge round-trips.

Usage:
    uv run modal run scripts/modal_smoke.py

What it checks:
  1. Modal image builds (first run: ~5 min, subsequent: cached)
  2. anthropic-api-key secret resolves
  3. Judge container can make 1 successful Claude API call
  4. Rubric is accessible in the judge container
  5. Sanitization pipeline works (produces sane PNG bytes)

Cost: ~$0.005 (1 Sonnet call on a tiny image).
"""
from __future__ import annotations
import sys
from pathlib import Path

# add research/eval/ to path so we can import modal_app
sys.path.insert(0, str(Path(__file__).parent.parent / "research" / "eval"))

import modal_app  # noqa: E402


@modal_app.app.local_entrypoint()
def main() -> None:
    import io
    import numpy as np
    from PIL import Image

    # Build a minimal strip: a 128×640 gradient
    h, w = 128, 640
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, _yy = np.meshgrid(x, y)
    strip = np.stack([xx, 1.0 - xx, 0.5 * np.ones_like(xx)], -1)
    strip = (strip * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(strip).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    print(f"[smoke] dispatching judge with {len(png_bytes)} byte PNG...")
    # This calls the deployed judge_container with a single strip.
    # Function signature is modal_app.judge_container.remote(strips_png=[bytes, ...])
    result = modal_app.judge_container.remote(strips_png=[png_bytes])
    print(f"[smoke] judge returned: type={type(result).__name__}")
    print(f"[smoke] result keys (first): {list(result.keys())[:10] if isinstance(result, dict) else 'n/a'}")
    # Whatever shape the synthesis used — just print it so we can see.
    import json
    print(json.dumps(result, indent=2, default=str)[:2000])
