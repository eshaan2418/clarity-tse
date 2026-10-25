"""Pre-render the training mixture pool (fp16 shards of 500 x 4 s binaural mixtures)."""
import argparse
import time

from clarity_tse.synth.train_data import build_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    t = time.time()
    files = build_pool(a.split, a.shards, a.jobs)
    print(f"{len(files)} shards in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
