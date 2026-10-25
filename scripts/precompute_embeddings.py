"""Precompute per-utterance ECAPA embeddings for training/val/test splits."""
import sys
import time

from clarity_tse.embed import speaker


def main() -> None:
    import os

    dev = os.environ.get("EMB_DEVICE", "cpu")
    for split in sys.argv[1:] or ["train", "val", "test"]:
        t = time.time()
        tab = speaker.utterance_embedding_table(split, device=dev)
        print(split, len(tab), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
