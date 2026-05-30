"""Pipeline entry point for the Basal-Bolus vs PPO comparison."""

import argparse

import evaluate
import inject_ppo_reference
import plots
import train


def run_full(args):
    if args.with_train:
        print("=== Step 1: Training PPO models ===")
        train.main()

    print("=== Step 2: Evaluating BB (and PPO if models exist) ===")
    evaluate.main()

    print("=== Step 3: Injecting PPO literature baseline ===")
    inject_ppo_reference.main()

    print("=== Step 4: Generating figures ===")
    plots.main()

    print("\nDone. Results in results/ and figures in results/figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Basal-Bolus vs PPO insulin control — pipeline runner"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train",    action="store_true", help="Train PPO models (slow, CPU-heavy)")
    group.add_argument("--inject",   action="store_true", help="Inject PPO literature baseline into results.csv")
    group.add_argument("--evaluate", action="store_true", help="Run evaluation and save results.csv")
    group.add_argument("--plots",    action="store_true", help="Generate all figures from results.csv")
    parser.add_argument(
        "--with-train",
        action="store_true",
        help="When running the full pipeline, also train PPO first (slow)",
    )

    args = parser.parse_args()

    if args.train:
        train.main()
    elif args.inject:
        inject_ppo_reference.main()
    elif args.evaluate:
        evaluate.main()
    elif args.plots:
        plots.main()
    else:
        run_full(args)


if __name__ == "__main__":
    main()
