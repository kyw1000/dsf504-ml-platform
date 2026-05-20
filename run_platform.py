"""
run_platform.py
===============
DSF504 ML Platform — Unified 6-Phase Runner

Usage
-----
    # Full pipeline for a single use case
    python run_platform.py --use-case A --steps all

    # Individual phases by number
    python run_platform.py --use-case A --steps 1
    python run_platform.py --use-case A --steps 1,2,3

    # Individual phases by name
    python run_platform.py --use-case A --steps data
    python run_platform.py --use-case A --steps eda
    python run_platform.py --use-case A --steps features
    python run_platform.py --use-case A --steps train
    python run_platform.py --use-case A --steps tune
    python run_platform.py --use-case A --steps ethics

    # Run all use cases for a single phase
    python run_platform.py --all-cases --steps 1

    # Run ALL use cases through ALL 6 phases
    python run_platform.py --all-cases --steps all

    # Launch the Streamlit dashboard
    python run_platform.py --dashboard

Use Cases
---------
    A        -> use_case_A_fraud          (Fraud Detection)
    B        -> use_case_B_credit         (Credit Scoring)
    C_nlp    -> use_case_C_nlp            (NLP Sentiment)
    C_market -> use_case_C_market         (Market Volatility)
    D        -> use_case_D_churn          (Customer Churn)
    E        -> use_case_E_insurance      (Insurance Risk)

6 Phases
--------
    1 / data      -> 01_data_loading.py
    2 / eda       -> 02_eda_analysis.py
    3 / features  -> 03_feature_engineering.py
    4 / train     -> 04_model_training.py
    5 / tune      -> 05_hyperparameter_tuning.py
    6 / ethics    -> 06_ethics_explainability.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# ── Use case registry (6 phases each) ────────────────────────────────────────

USE_CASE_SCRIPTS: dict[str, dict[int, str]] = {
    "A": {
        1: "use_case_A_fraud/01_data_loading.py",
        2: "use_case_A_fraud/02_eda_analysis.py",
        3: "use_case_A_fraud/03_feature_engineering.py",
        4: "use_case_A_fraud/04_model_training.py",
        5: "use_case_A_fraud/05_hyperparameter_tuning.py",
        6: "use_case_A_fraud/06_ethics_explainability.py",
    },
    "B": {
        1: "use_case_B_credit/01_data_loading.py",
        2: "use_case_B_credit/02_eda_analysis.py",
        3: "use_case_B_credit/03_feature_engineering.py",
        4: "use_case_B_credit/04_model_training.py",
        5: "use_case_B_credit/05_hyperparameter_tuning.py",
        6: "use_case_B_credit/06_ethics_explainability.py",
    },
    "C_nlp": {
        1: "use_case_C_nlp/01_data_loading.py",
        2: "use_case_C_nlp/02_eda_analysis.py",
        3: "use_case_C_nlp/03_feature_engineering.py",
        4: "use_case_C_nlp/04_model_training.py",
        5: "use_case_C_nlp/05_hyperparameter_tuning.py",
        6: "use_case_C_nlp/06_ethics_explainability.py",
    },
    "C_market": {
        1: "use_case_C_market/01_data_loading.py",
        2: "use_case_C_market/02_eda_analysis.py",
        3: "use_case_C_market/03_feature_engineering.py",
        4: "use_case_C_market/04_model_training.py",
        5: "use_case_C_market/05_hyperparameter_tuning.py",
        6: "use_case_C_market/06_ethics_explainability.py",
    },
    "D": {
        1: "use_case_D_churn/01_data_loading.py",
        2: "use_case_D_churn/02_eda_analysis.py",
        3: "use_case_D_churn/03_feature_engineering.py",
        4: "use_case_D_churn/04_model_training.py",
        5: "use_case_D_churn/05_hyperparameter_tuning.py",
        6: "use_case_D_churn/06_ethics_explainability.py",
    },
    "E": {
        1: "use_case_E_insurance/01_data_loading.py",
        2: "use_case_E_insurance/02_eda_analysis.py",
        3: "use_case_E_insurance/03_feature_engineering.py",
        4: "use_case_E_insurance/04_model_training.py",
        5: "use_case_E_insurance/05_hyperparameter_tuning.py",
        6: "use_case_E_insurance/06_ethics_explainability.py",
    },
}

USE_CASE_LABELS: dict[str, str] = {
    "A":        "Fraud Detection",
    "B":        "Credit Scoring",
    "C_nlp":    "NLP Sentiment",
    "C_market": "Market Volatility",
    "D":        "Customer Churn",
    "E":        "Insurance Risk",
}

PHASE_LABELS: dict[int, str] = {
    1: "Data Loading",
    2: "EDA Analysis",
    3: "Feature Engineering",
    4: "Model Training",
    5: "Hyperparameter Tuning",
    6: "Ethics & Explainability",
}

STEP_ALIASES: dict[str, int] = {
    "data":           1,
    "load":           1,
    "eda":            2,
    "explore":        2,
    "features":       3,
    "feature":        3,
    "fe":             3,
    "train":          4,
    "model":          4,
    "tune":           5,
    "tuning":         5,
    "hp":             5,
    "hpo":            5,
    "ethics":         6,
    "explainability": 6,
    "shap":           6,
    "xai":            6,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(char: str = "=", width: int = 64) -> str:
    return char * width


def resolve_steps(steps_arg: str, uc: str) -> list[int]:
    scripts = USE_CASE_SCRIPTS.get(uc, {})
    if steps_arg.strip().lower() == "all":
        return sorted(scripts.keys())
    result: list[int] = []
    for token in steps_arg.split(","):
        token = token.strip().lower()
        if token in STEP_ALIASES:
            result.append(STEP_ALIASES[token])
        elif token.isdigit():
            n = int(token)
            if n in scripts:
                result.append(n)
            else:
                print(f"[warn] Phase {n} is not defined for Use Case {uc} — skipping.")
        else:
            print(f"[warn] Unknown step '{token}' — skipping.")
    return sorted(set(result))


def run_script(script_path: Path, phase: int, uc: str) -> int:
    label = PHASE_LABELS.get(phase, f"Phase {phase}")
    uc_label = USE_CASE_LABELS.get(uc, uc)
    print(f"\n{_sep()}")
    print(f"  Use Case {uc}: {uc_label}")
    print(f"  Phase {phase}: {label}")
    print(f"  Script : {script_path.relative_to(ROOT)}")
    print(_sep())
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "✓ OK" if result.returncode == 0 else "✗ FAILED"
    print(f"\n  {status}  ({elapsed:.1f}s)")
    return result.returncode


def launch_dashboard() -> None:
    print(f"\n{_sep()}")
    print("  Launching Streamlit dashboard")
    print("  URL: http://localhost:8501")
    print(_sep())
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.headless", "false"],
        cwd=str(ROOT),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSF504 ML Platform — 6-Phase Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--use-case", "-u", default="A",
        help="Use case key: A | B | C_nlp | C_market | D | E  (default: A)",
    )
    parser.add_argument(
        "--all-cases", "-a", action="store_true",
        help="Run across ALL active use cases",
    )
    parser.add_argument(
        "--steps", "-s", default="all",
        help=(
            "Phases to run: all | 1-6 | data | eda | features | train | tune | ethics"
            " | comma-separated list, e.g. 1,2,3"
        ),
    )
    parser.add_argument(
        "--dashboard", "-d", action="store_true",
        help="Launch the Streamlit dashboard instead of running a pipeline",
    )
    parser.add_argument(
        "--continue-on-error", "-c", action="store_true",
        help="Continue to the next phase even if one fails (default: stop on error)",
    )
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
        return

    # Determine which use cases to run
    if args.all_cases:
        uc_keys = list(USE_CASE_SCRIPTS.keys())
    else:
        uc = args.use_case
        if "_" not in uc:
            uc = uc.upper()
        if uc not in USE_CASE_SCRIPTS:
            valid = ", ".join(USE_CASE_SCRIPTS.keys())
            print(f"[error] Unknown use case '{uc}'. Valid keys: {valid}")
            sys.exit(1)
        uc_keys = [uc]

    # Summary header
    print(f"\n{_sep()}")
    print("  DSF504 ML Platform — 6-Phase Runner")
    print(f"  Use cases : {', '.join(uc_keys)}")
    print(f"  Phases    : {args.steps}")
    print(_sep())

    overall_errors: list[tuple[str, int]] = []   # (uc_key, phase)

    for uc in uc_keys:
        steps = resolve_steps(args.steps, uc)
        if not steps:
            print(f"[warn] No valid phases resolved for Use Case {uc} — skipping.")
            continue

        uc_errors: list[int] = []
        for phase in steps:
            script_rel = USE_CASE_SCRIPTS[uc].get(phase)
            if script_rel is None:
                print(f"[warn] Phase {phase} not defined for Use Case {uc}.")
                continue
            script_path = ROOT / script_rel
            if not script_path.exists():
                print(f"[error] Script not found: {script_path}")
                overall_errors.append((uc, phase))
                if not args.continue_on_error:
                    break
                continue
            rc = run_script(script_path, phase, uc)
            if rc != 0:
                overall_errors.append((uc, phase))
                uc_errors.append(phase)
                if not args.continue_on_error:
                    break

        if uc_errors and not args.continue_on_error:
            break

    # Final report
    print(f"\n{_sep()}")
    if overall_errors:
        print("  RESULT: FAILED")
        for uc, phase in overall_errors:
            label = PHASE_LABELS.get(phase, f"Phase {phase}")
            print(f"    ✗ Use Case {uc} — Phase {phase}: {label}")
        print(_sep())
        sys.exit(1)
    else:
        print("  RESULT: All phases completed successfully ✓")
        print(_sep())
        print("\nNext step — launch the dashboard:")
        print("    python run_platform.py --dashboard\n")


if __name__ == "__main__":
    main()
