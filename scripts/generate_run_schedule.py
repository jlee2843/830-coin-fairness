import argparse
import csv
import random
from itertools import product
from pathlib import Path


DENOMINATIONS = ["Nickel", "Dime", "Quarter"]
DECADES = ["1980s", "2000s", "2010s"]
POSTURES = ["Standing", "Sitting"]
FLIPPERS = ["Jenny", "Josh", "Esther"]
STARTING_SIDES = ["Heads", "Tails"]

DEFAULT_OUTPUT = Path("data/run_schedule.csv")


def build_schedule(replications, seed):
    rows = []
    rng = random.Random(seed)

    for replication in range(1, replications + 1):
        replication_rows = []

        for denomination, decade, posture, flipper, starting_side in product(
            DENOMINATIONS,
            DECADES,
            POSTURES,
            FLIPPERS,
            STARTING_SIDES,
        ):
            picker, recorder = rng.sample(
                [person for person in FLIPPERS if person != flipper],
                k=2,
            )

            replication_rows.append({
                "replication": replication,
                "denomination": denomination,
                "decade": decade,
                "posture": posture,
                "flipper": flipper,
                "picker": picker,
                "recorder": recorder,
                "starting_side": starting_side,
            })

        rng.shuffle(replication_rows)
        rows.extend(replication_rows)

    for run_id, row in enumerate(rows, start=1):
        row["run_id"] = run_id

    return rows


def write_schedule(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "replication",
        "denomination",
        "decade",
        "posture",
        "flipper",
        "picker",
        "recorder",
        "starting_side",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a randomized full run schedule for the coin experiment."
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=3,
        help="Number of replications per full factorial combination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=830,
        help="Random seed used to make the schedule reproducible.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path for the generated schedule.",
    )

    args = parser.parse_args()

    if args.replications < 1:
        raise ValueError("replications must be at least 1")

    rows = build_schedule(args.replications, args.seed)
    write_schedule(rows, args.output)

    print(f"Wrote {len(rows)} randomized runs to {args.output}")


if __name__ == "__main__":
    main()
