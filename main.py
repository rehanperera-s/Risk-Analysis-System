"""
Main pipeline runner for the Betting Platform Customer Churn Prediction project.

Usage:
    python main.py              # Run full pipeline (data + model)
    python main.py --data-only  # Only fetch data and build features
    python main.py --model-only # Only train models (requires existing data)
"""

import sys

from src.data_collection import run_data_pipeline
from src.model_training import run_training_pipeline


def main():
    args = sys.argv[1:]

    if "--model-only" in args:
        print("Running model training only...\n")
        run_training_pipeline()
    elif "--data-only" in args:
        print("Running data pipeline only...\n")
        run_data_pipeline()
    else:
        print("Running full pipeline...\n")
        run_data_pipeline()
        print("\n" + "=" * 60 + "\n")
        run_training_pipeline()


if __name__ == "__main__":
    main()
