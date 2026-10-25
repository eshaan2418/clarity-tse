"""Build the training RIR bank (static-head geometries from the train split)."""
import argparse
import time

from clarity_tse.synth.train_data import build_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    t = time.time()
    files = build_bank(a.split, a.n, a.jobs, max_order_cap=60)
    print(f"{len(files)} entries in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
