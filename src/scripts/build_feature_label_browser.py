#!/usr/bin/env python3
"""
Build a lightweight local HTML browser for feature labels JSONL.

This viewer does NOT load model weights and does NOT run SAE inference.
It is intended for browsing already-generated labels quickly.

Examples:
  uv run -m src.scripts.build_feature_label_browser
  uv run -m src.scripts.build_feature_label_browser --input-jsonl data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl --output-html data/neuron_labels_mamay/results/feature_label_browser.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build static HTML browser for feature label JSONL.")
    p.add_argument(
        "--input-jsonl",
        default="data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl",
        help="Input JSONL with feature labels.",
    )
    p.add_argument(
        "--output-html",
        default="data/neuron_labels_mamay/results/feature_label_browser.html",
        help="Output HTML path.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Max rows to include in browser payload.",
    )
    return p.parse_args()


def _load_rows(path: Path, max_rows: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _build_html(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MamayScope Feature Label Browser</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0b0f14; color: #e6edf3; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 16px; }}
    .controls {{ display: grid; gap: 8px; grid-template-columns: 1fr 220px 220px; margin-bottom: 12px; }}
    input, select {{ background: #111827; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }}
    .meta {{ color: #8b949e; margin-bottom: 10px; }}
    .card {{ border: 1px solid #30363d; border-radius: 10px; padding: 12px; margin-bottom: 10px; background: #111827; }}
    .head {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
    .pill {{ border: 1px solid #30363d; border-radius: 999px; padding: 2px 10px; color: #9ecbff; }}
    .label {{ font-size: 18px; font-weight: 700; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: #9ecbff; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }}
    ul {{ margin: 8px 0 0 16px; padding: 0; }}
    li {{ margin-bottom: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>MamayScope Feature Label Browser</h2>
    <div class="controls">
      <input id="q" placeholder="Search label, feature id, thought process, contexts..." />
      <select id="sort">
        <option value="feature_id">Sort: feature_id</option>
        <option value="sampled_desc">Sort: sampled_from_total (desc)</option>
        <option value="label">Sort: label (A-Z)</option>
      </select>
      <select id="limit">
        <option value="20">Show 20</option>
        <option value="50" selected>Show 50</option>
        <option value="100">Show 100</option>
        <option value="200">Show 200</option>
      </select>
    </div>
    <div class="meta" id="meta"></div>
    <div id="list"></div>
  </div>
  <script>
    const data = {payload};
    const qEl = document.getElementById("q");
    const sortEl = document.getElementById("sort");
    const limitEl = document.getElementById("limit");
    const listEl = document.getElementById("list");
    const metaEl = document.getElementById("meta");

    function txt(x) {{ return (x ?? "").toString().toLowerCase(); }}
    function scoreText(r) {{
      return [
        r.feature_id, r.label, r.thought_process, r.neuronpedia_feature_id,
        ...(Array.isArray(r.top_contexts) ? r.top_contexts : [])
      ].join(" ").toLowerCase();
    }}

    function render() {{
      const q = txt(qEl.value).trim();
      let rows = data.filter(r => !q || scoreText(r).includes(q));

      const mode = sortEl.value;
      if (mode === "feature_id") rows.sort((a,b) => (a.feature_id ?? 0) - (b.feature_id ?? 0));
      if (mode === "sampled_desc") rows.sort((a,b) => (b.sampled_from_total ?? 0) - (a.sampled_from_total ?? 0));
      if (mode === "label") rows.sort((a,b) => txt(a.label).localeCompare(txt(b.label)));

      const limit = parseInt(limitEl.value, 10) || 50;
      const shown = rows.slice(0, limit);
      metaEl.textContent = `Total loaded: ${{data.length}} | Matched: ${{rows.length}} | Showing: ${{shown.length}}`;

      listEl.innerHTML = shown.map(r => {{
        const contexts = Array.isArray(r.top_contexts) ? r.top_contexts.slice(0, 20) : [];
        return `
          <div class="card">
            <div class="head">
              <span class="pill">feature #${{r.feature_id}}</span>
              <span class="pill">sampled: ${{r.sampled_from_total ?? "n/a"}}</span>
            </div>
            <div class="label">${{r.label ?? "(no label)"}}</div>
            <details>
              <summary>thought_process</summary>
              <pre>${{(r.thought_process ?? "").replace(/</g, "&lt;")}}</pre>
            </details>
            <details>
              <summary>top_contexts (${{contexts.length}})</summary>
              <ul>${{contexts.map(c => `<li>${{(c ?? "").replace(/</g, "&lt;")}}</li>`).join("")}}</ul>
            </details>
          </div>
        `;
      }}).join("");
    }}

    qEl.addEventListener("input", render);
    sortEl.addEventListener("change", render);
    limitEl.addEventListener("change", render);
    render();
  </script>
</body>
</html>"""


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_path, args.max_rows)
    html = _build_html(rows)
    output_path.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "input_jsonl": str(input_path),
                "rows_included": len(rows),
                "output_html": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
