"""
run_platform.py
===============
DSF504 ML Platform — Unified runner

Usage
-----
    # Full pipeline for Use Case A (fraud)
    python run_platform.py --use-case A --steps all

    # Individual steps
    python run_platform.py --use-case A --steps data
    python run_platform.py --use-case A --steps eda
    python run_platform.py --use-case A --steps features
    python run_platform.py --use-case A --steps train
    python run_platform.py --use-case A --steps tune
    python run_platform.py --use-case A --steps 1,2,3

    # Launch dashboard
    python run_platform.py --dashboard

Steps
-----
1 / data      -> use_case_A_fraud/01_data_loading.py
2 / eda       -> use_case_A_fraud/02_eda_analysis.py
3 / features  -> use_case_A_fraud/03_feature_engineering.py
4 / train     -> use_case_A_fraud/04_model_training.py
5 / tune      -> use_case_A_fraud/05_hyperparameter_tuning.py
dashboard     -> streamlit run dashboard/app.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

USE_CASE_SCRIPTS = {
    "A": {
        1: "use_case_A_fraud/01_data_loading.py",
        2: "use_case_A_fraud/02_eda_analysis.py",
        3: "use_case_A_fraud/03_feature_engineering.py",
        4: "use_case_A_fraud/04_model_training.py",
        5: "use_case_A_fraud/05_hyperparameter_tuning.py",
    },
    "B": {
        1: "use_case_B_credit/01_data_loading.py",
        2: "use_case_B_credit/02_eda_analysis.py",
        3: "use_case_B_credit/03_feature_engineering.py",
        4: "use_case_B_credit/04_model_training.py",
        5: "use_case_B_credit/05_hyperparameter_tuning.py",
    },
    "C_nlp": {
        1: "use_case_C_nlp/01_data_loading.py",
        2: "use_case_C_nlp/02_eda_analysis.py",
        3: "use_case_C_nlp/03_feature_engineering.py",
        4: "use_case_C_nlp/04_model_training.py",
        5: "use_case_C_nlp/05_hyperparameter_tuning.py",
    },
    # Add C_markets, D-G as implemented
}

STEP_ALIASES = {
    "data":     1,
    "eda":      2,
    "features": 3,
    "feature":  3,
    "train":    4,
    "tune":     5,
    "tuning":   5,
}


def resolve_steps(steps_arg: str, uc: str) -> list:
    scripts = USE_CASE_SCRIPTS.get(uc, {})
    if steps_arg == "all":
        return sorted(scripts.keys())
    result = []
    for token in steps_arg.split(","):
        token = token.strip()
        if token in STEP_ALIASES:
            result.append(STEP_ALIASES[token])
        elif token.isdigit():
            result.append(int(token))
        else:
            print(f"[warn] Unknown step '{token}' -- skipping.")
    return sorted(set(result))


def run_script(script_path: Path) -> int:
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
    return result.returncode


def launch_dashboard() -> None:
    print("\nLaunching Streamlit dashboard...")
    print("Open: http://localhost:8501\n")
    subprocess.run(
        ["streamlit", "run", "dashboard/app.py",
         "--server.headless", "false"],
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSF504 ML Platform runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--use-case", "-u", default="A",
                        help="Use case key: A, B, C_nlp, C_markets, D, E, F, G")
    parser.add_argument("--steps", "-s", default="all",
                        help="Steps to run: all | data | eda | features | train | tune | 1,2,3")
    parser.add_argument("--dashboard", "-d", action="store_true",
                        help="Launch the Streamlit dashboard")
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
        return

    uc = args.use_case
    # Normalise: plain letters become uppercase, underscored keys preserved
    if "_" not in uc:
        uc = uc.upper()

    if uc not in USE_CASE_SCRIPTS:
        print(f"Error: Unknown use case '{uc}'. Valid: {list(USE_CASE_SCRIPTS.keys())}")
        sys.exit(1)

    steps = resolve_steps(args.steps, uc)
    if not steps:
        print("No valid steps specified.")
        sys.exit(1)

    print(f"\nDSF504 ML Platform -- Use Case {uc}")
    print(f"Running steps: {steps}")

    errors = []
    for step in steps:
        script_rel = USE_CASE_SCRIPTS[uc].get(step)
        if script_rel is None:
            print(f"[warn] Step {step} not defined for Use Case {uc}.")
            continue
        script_path = ROOT / script_rel
        if not script_path.exists():
            print(f"[error] Script not found: {script_path}")
            errors.append(step)
            continue
        rc = run_script(script_path)
        if rc != 0:
            print(f"[error] Step {step} failed (exit code {rc}).")
            errors.append(step)
            break  # stop on failure

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED steps: {errors}")
        sys.exit(1)
    else:
        print("All steps completed successfully.")
        print("  -> Launch dashboard: python run_platform.py --dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
