"""Tuning script: Hill-climb / random search over evaluation function weights.

Optimizes parameters in agent/weights.json using parallel win-rate evaluation against F.
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.peval import summarize, _run_chunk
from catan_kit import match
from concurrent.futures import ProcessPoolExecutor, as_completed


WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "weights.json")

# Default parameter space to explore
TUNABLE_KEYS = {
    "road_progress": [2e2, 5e2, 1e3, 2e3, 5e3],
    "army_progress": [2e2, 5e2, 1e3, 2e3, 5e3],
    "city_plan": [1e3, 5e3, 2e4, 1e5],
    "port_value": [1e6, 3e6, 5e6, 1e7],
    "discard_risk": [-10, -20, -50, -100],
    "opp_dev_vp": [-1e5, -3e5, -1e6],
}


def load_current_weights():
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            return json.load(f)
    from agent.evalfn import W
    return copy.deepcopy(W)


def save_weights(w):
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(w, f, indent=2)


def eval_weights(weights, games=100, seed0=990000, workers=16):
    save_weights(weights)
    seeds = [seed0 + i // 2 for i in range(games)]
    chunks = []
    raw_per = max(4, games // (workers * 2))
    per = max(2, (raw_per // 2) * 2)
    for i in range(0, len(seeds), per):
        chunk_seeds = seeds[i:i + per]
        chunk_indices = list(range(i, i + len(chunk_seeds)))
        chunks.append(("agent/agent.py", "examples/f_agent.py", chunk_seeds, chunk_indices, match.DEFAULT_KEY))

    records = []
    n_workers = min(workers, len(chunks))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_run_chunk, c) for c in chunks]
        for f in as_completed(futs):
            recs, _ = f.result()
            records.extend(recs)

    res = summarize(records)
    return res["win_rate"], res["ci95"], res["mean_cpu_s"], res["games_over_budget"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed0", type=int, default=990000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--iterations", type=int, default=10)
    args = p.parse_args()

    current_w = load_current_weights()
    print(f"Initial evaluation of baseline weights over {args.games} games...")
    best_wr, ci, cpu, over = eval_weights(current_w, games=args.games, seed0=args.seed0, workers=args.workers)
    print(f"Baseline Win Rate: {best_wr:.4f} +- {ci:.4f} (CPU: {cpu*1000:.0f} ms, over: {over})")

    for it in range(args.iterations):
        import random
        key = random.choice(list(TUNABLE_KEYS.keys()))
        cand_val = random.choice(TUNABLE_KEYS[key])
        if cand_val == current_w.get(key):
            continue

        cand_w = copy.deepcopy(current_w)
        cand_w[key] = cand_val
        print(f"\n[Iter {it+1}/{args.iterations}] Testing {key} = {cand_val} (was {current_w.get(key)})...")
        wr, ci, cpu, over = eval_weights(cand_w, games=args.games, seed0=args.seed0, workers=args.workers)
        print(f"Result: {wr:.4f} +- {ci:.4f} (CPU: {cpu*1000:.0f} ms, over: {over})")

        # Keep only clear improvements
        if wr > best_wr and over == 0:
            print(f"--> ACCEPTED! Win rate improved from {best_wr:.4f} to {wr:.4f}")
            best_wr = wr
            current_w = cand_w
            save_weights(current_w)
        else:
            print(f"--> Rejected (best remains {best_wr:.4f})")
            save_weights(current_w)

    print("\nTuning finished. Final best weights saved to agent/weights.json.")


if __name__ == "__main__":
    main()
