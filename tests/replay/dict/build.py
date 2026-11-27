"""Build the project's own dictionary on the whole material and report its size, build time and
sample links. `python -I tests/replay/dict/build.py [--alpha 0.01] [--out scratch/lead-dict/dict.json]`"""
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lexicon", HERE / "lexicon.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def main(argv):
    alpha = float(argv[argv.index("--alpha") + 1]) if "--alpha" in argv else 0.01
    corr = argv[argv.index("--correction") + 1] if "--correction" in argv else "bh"
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else L.ROOT / "scratch" / "lead-dict" / "dict.json"
    t0 = time.perf_counter()
    files = L.material_files()
    t1 = time.perf_counter()
    items = [it for rel, lines in files.items() for it in L.doc_items(rel, lines)]
    pl = L.pairs(files, items)
    t2 = time.perf_counter()
    links, info = L.build_dictionary(pl, alpha=alpha, correction=corr, strata="--strata" in argv)
    t3 = time.perf_counter()
    print(f"files {len(files)}, documented items {len(items)} by kind {dict(Counter(i['kind'] for i in items))}")
    print(f"dictionary {info}")
    print(f"time: read {t1 - t0:.2f} s, pair {t2 - t1:.2f} s, count and test {t3 - t2:.2f} s, total {t3 - t0:.2f} s")
    per = sorted(len(v) for v in links.values())
    print(f"links per linked word: median {per[len(per) // 2]}, 90th {per[int(len(per) * 0.9)]}, max {per[-1]}")
    rng = random.Random(L.SEED)
    words = sorted(links)
    print("20 sample links (word -> strongest parts, strength):")
    for w in rng.sample(words, min(20, len(words))):
        top = sorted(links[w].items(), key=lambda x: -x[1])[:4]
        print(f"  {w} -> " + ", ".join(f"{c} {s:.2f}" for c, s in top))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alpha": alpha, "correction": corr, "info": info, "links": links}), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
