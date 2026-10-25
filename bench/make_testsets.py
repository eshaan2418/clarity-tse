"""M1: generate frozen test/val sets with checksummed manifests."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

from clarity_tse.eval.latency import provenance
from clarity_tse.synth import corpus, testsets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["test_main", "val_main"])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    for name in args.sets:
        specs = testsets.specs_for(name)
        with Pool(args.jobs) as p:
            items = p.map(testsets.render_one, [(name, s) for s in specs], chunksize=1)
        testsets.manifest_dir(name).mkdir(parents=True, exist_ok=True)
        man = dict(name=name, definition=testsets.SET_DEFS[name], n_items=len(items),
                   esc50_available=bool(corpus.esc50_index("test")), items=items,
                   provenance=provenance(config=testsets.SET_DEFS[name]))
        (testsets.manifest_dir(name) / f"{name}.json").write_text(json.dumps(man, indent=1))
        print(name, len(items), "items")


if __name__ == "__main__":
    main()
