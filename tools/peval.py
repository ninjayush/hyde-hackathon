"""Parallel evaluator: same seeds/seat-pairing as catan_kit.evaluate, split across processes.

    python tools/peval.py --a agent/agent.py --b examples/f_agent.py --games 400

Appends one JSON line per run to results/runs.jsonl.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan_kit import match


def _run_chunk(args):
    path_a, path_b, seeds, indices, key = args
    records, info = match.play_chunk(path_a, path_b, seeds, indices, key=key)
    return records, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed0", type=int, default=990000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--label", default="")
    args = p.parse_args()

    # match.play_chunk uses the same seed->pair mapping as evaluate.run
    seeds = [args.seed0 + i // 2 for i in range(args.games)]
    chunks = []
    # Ensure per is a multiple of 2 so every chunk has balanced colors on identical seeds
    raw_per = max(4, args.games // (args.workers * 2))
    per = max(2, (raw_per // 2) * 2)
    for i in range(0, len(seeds), per):
        chunk_seeds = seeds[i:i + per]
        chunk_indices = list(range(i, i + len(chunk_seeds)))
        chunks.append((args.a, args.b, chunk_seeds, chunk_indices, match.DEFAULT_KEY))

    t0 = time.time()
    records = []
    n_workers = min(args.workers, len(chunks))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_run_chunk, c) for c in chunks]
        for f in as_completed(futs):
            recs, _ = f.result()
            records.extend(recs)

    res = summarize(records)
    res.update(a=args.a, b=args.b, games=args.games, label=args.label,
               wall_s=round(time.time() - t0, 1), workers=n_workers)
    os.makedirs("results", exist_ok=True)
    with open("results/runs.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")

    fs = res["first_seat"]
    fsw = res["first_seat_wins"]
    print(f"A = {args.a}  vs  B = {args.b}")
    print(f"win rate {res['win_rate']:.4f} +-{res['ci95']:.4f}  ({res['wins']}/{res['games']})"
          f"  first-seat {fsw}/{fs}")
    print(f"no-winner {res['no_winner']}  mean CPU {res['mean_cpu_s']*1000:.0f} ms"
          f"  worst {res['worst_cpu_s']*1000:.0f} ms  over {res['games_over_budget']}"
          f"  faults {res['errors']}+{res['timeouts']}+{res['crashes']}")
    print(f"wall {res['wall_s']} s  ({len(records)/res['wall_s']:.1f} games/s, {n_workers} workers)")


def summarize(records, side="a"):
    n = len(records)
    me = [r[side] for r in records]
    wins = sum(r["winner"] == side for r in records)
    rate = wins / n
    cpu = [m["cpu_s"] for m in me]
    return dict(games=n, wins=wins, win_rate=rate, ci95=1.96 * (rate * (1 - rate) / n) ** 0.5,
                no_winner=sum(r["winner"] is None for r in records),
                mean_cpu_s=sum(cpu) / n, worst_cpu_s=max(cpu),
                games_over_budget=sum(m["over"] is not None for m in me),
                errors=sum(m["errors"] for m in me),
                timeouts=sum(m["timeouts"] for m in me),
                crashes=sum(m["crashes"] for m in me),
                first_seat_wins=sum(1 for r in records
                                    if r["winner"] == side and r["a_first"]),
                first_seat=sum(1 for r in records if r["a_first"]),
                )


if __name__ == "__main__":
    # first-seat tracking: record dict may not carry it; compute from pair order instead
    main()
