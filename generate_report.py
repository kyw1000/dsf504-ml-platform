"""
generate_report.py
==================
Generates a self-contained HTML report for Use Case A by embedding all
PNG plots as base64 data URIs.  Open the output file in the Claude right
panel — no local-file-loading issues.

Usage
-----
    python generate_report.py
    # Output: C:/DSF504/reports/use_case_A/report.html
"""

import base64
import sys
from pathlib import Path
from datetime import datetime

ROOT       = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "use_case_A"
OUT_HTML   = REPORT_DIR / "report.html"

# ── Plot catalogue (label → filename, in display order) ──────────────────────
SECTIONS = [
    ("📊 Data Overview", [
        ("Target Distribution (Fraud vs Legit)", "target_distribution.png"),
        ("Missing Values by Group",              "missing_by_group.png"),
        ("Missing Values Heatmap",               "missing_heatmap.png"),
        ("Transaction Amount Distribution",      "transaction_amount_distribution.png"),
        ("Time Patterns",                        "time_patterns.png"),
    ]),
    ("🔍 EDA — Feature Analysis", [
        ("Categorical Feature Fraud Rates",      "categorical_fraud_rates.png"),
        ("C-Columns by Fraud Label",             "C_columns_by_fraud.png"),
        ("D-Columns by Fraud Label",             "D_columns_by_fraud.png"),
        ("Correlation Heatmap (C-cols)",         "correlation_heatmap_C_cols.png"),
        ("Correlation Top-30 V-cols",            "correlation_top30_V_cols.png"),
    ]),
    ("⚙️ Feature Engineering", [
        ("Engineered Feature Summary",           "engineered_feature_summary.png"),
    ]),
    ("🤖 Model Training", [
        ("ROC & PR Curves",                      "roc_pr_curves.png"),
        ("Confusion Matrices",                   "confusion_matrices.png"),
        ("CV Model Comparison",                  "cv_comparison.png"),
    ]),
    ("🎯 Hyperparameter Tuning", [
        ("Tuned vs Untuned Comparison",          "tuned_vs_untuned_comparison.png"),
        ("Optuna Trial History",                 "optuna_history.png"),
        ("Champion Threshold Calibration",       "champion_threshold_calibration.png"),
        ("SHAP Feature Importance (Bar)",        "shap_bar_importance.png"),
        ("SHAP Beeswarm (Direction)",            "shap_beeswarm.png"),
    ]),
]


def b64_img(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def build_html() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections_html = ""
    total_found = 0
    total_missing = 0

    for section_title, plots in SECTIONS:
        cards = ""
        for label, fname in plots:
            src = b64_img(REPORT_DIR / fname)
            if src:
                total_found += 1
                cards += f"""
                <div class="card">
                  <p class="card-label">{label}</p>
                  <img src="{src}" alt="{label}" loading="lazy">
                </div>"""
            else:
                total_missing += 1
                cards += f"""
                <div class="card missing">
                  <p class="card-label">{label}</p>
                  <div class="placeholder">⏳ Not yet generated<br>
                    <small>{fname}</small></div>
                </div>"""

        sections_html += f"""
        <section>
          <h2>{section_title}</h2>
          <div class="grid">{cards}
          </div>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DSF504 — Use Case A Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e0e0e0;
    padding: 24px 32px 60px;
  }}
  header {{
    border-bottom: 2px solid #1E88E5;
    padding-bottom: 14px; margin-bottom: 32px;
  }}
  header h1 {{ font-size: 1.7rem; color: #1E88E5; }}
  header p  {{ font-size: 0.85rem; color: #888; margin-top: 4px; }}
  .badge {{
    display: inline-block;
    background: #1E88E522; color: #1E88E5;
    border-radius: 12px; padding: 3px 12px;
    font-size: 0.75rem; font-weight: 600;
    margin-left: 10px; vertical-align: middle;
  }}
  section {{ margin-bottom: 40px; }}
  section h2 {{
    font-size: 1.1rem; margin-bottom: 16px;
    padding: 6px 14px; background: #1a1d27;
    border-left: 4px solid #1E88E5; border-radius: 0 6px 6px 0;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
    gap: 20px;
  }}
  .card {{
    background: #1a1d27; border-radius: 10px;
    overflow: hidden; border: 1px solid #2a2d3a;
  }}
  .card img {{
    width: 100%; display: block;
  }}
  .card-label {{
    font-size: 0.78rem; color: #aaa;
    padding: 8px 12px 6px; font-weight: 500;
  }}
  .card.missing {{ opacity: 0.45; }}
  .placeholder {{
    height: 180px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: #666; font-size: 0.9rem; gap: 8px;
  }}
  .placeholder small {{ color: #444; font-family: monospace; }}
  footer {{
    margin-top: 48px; text-align: center;
    font-size: 0.75rem; color: #444;
    border-top: 1px solid #222; padding-top: 16px;
  }}
</style>
</head>
<body>
<header>
  <h1>🏦 DSF504 — Use Case A: IEEE Fraud Detection
    <span class="badge">Generated {ts}</span>
  </h1>
  <p>{total_found} plots embedded · {total_missing} pending (re-run after pipeline completes)</p>
</header>

{sections_html}

<footer>DSF504 ML Platform · Ko-Yang Wang · Fusions360</footer>
</body>
</html>"""


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"✅  Report written → {OUT_HTML}  ({size_kb:.0f} KB)")
    print(f"    Open in Claude right panel: computer://{OUT_HTML}")
