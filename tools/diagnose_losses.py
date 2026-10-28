"""Diagnostic tool: Replay and analyze games where agent A lost against agent B.

Extracts:
- Exact opening settlements (first 2 per player from action records)
- Total settlements and cities built throughout the game
- Largest Army and Longest Road outcomes
- End-game VPs (public and actual)
- Failure mode categorization
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan_kit import match
from catan_kit.match import Seat, AgentProcess, COLORS, DEFAULT_KEY, dice_seed
from catanatron import Game
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ActionType
from catanatron.features import get_node_production
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
)


def get_opening_from_records(game, color):
    cmap = game.state.board.map
    # First 4 settlement actions in the game are the opening placements
    settlement_actions = [
        r.action for r in game.state.action_records
        if r.action.action_type == ActionType.BUILD_SETTLEMENT
    ]
    my_opening_nodes = [a.value for a in settlement_actions[:4] if a.color == color]
    
    total_pips = 0.0
    res_pips = {r: 0.0 for r in RESOURCES}
    for n in my_opening_nodes:
        for r in RESOURCES:
            p = get_node_production(cmap, n, r) * 36.0
            total_pips += p
            res_pips[r] += p
    variety = sum(1 for p in res_pips.values() if p > 0.01)
    ow_pips = res_pips["ORE"] + res_pips["WHEAT"]
    return {
        "nodes": my_opening_nodes,
        "total_pips": round(total_pips, 1),
        "variety": variety,
        "ow_pips": round(ow_pips, 1),
        "res_pips": {k: round(v, 1) for k, v in res_pips.items()},
    }


def analyze_loss(game, color_a, color_b):
    st = game.state
    key_a = player_key(st, color_a)
    key_b = player_key(st, color_b)
    
    vp_a = st.player_state.get(f"{key_a}_ACTUAL_VICTORY_POINTS", 0)
    vp_b = st.player_state.get(f"{key_b}_ACTUAL_VICTORY_POINTS", 0)
    pub_vp_a = st.player_state.get(f"{key_a}_VICTORY_POINTS", 0)
    pub_vp_b = st.player_state.get(f"{key_b}_VICTORY_POINTS", 0)
    
    open_a = get_opening_from_records(game, color_a)
    open_b = get_opening_from_records(game, color_b)
    
    # Buildings built
    settlements_a = len(st.buildings_by_color[color_a][SETTLEMENT])
    cities_a = len(st.buildings_by_color[color_a][CITY])
    settlements_b = len(st.buildings_by_color[color_b][SETTLEMENT])
    cities_b = len(st.buildings_by_color[color_b][CITY])
    
    # Army and Road
    army_a = get_played_dev_cards(st, color_a, "KNIGHT")
    army_b = get_played_dev_cards(st, color_b, "KNIGHT")
    has_army_a = st.player_state.get(f"{key_a}_HAS_ARMY", False)
    has_army_b = st.player_state.get(f"{key_b}_HAS_ARMY", False)
    
    road_len_a = get_longest_road_length(st, color_a)
    road_len_b = get_longest_road_length(st, color_b)
    has_road_a = st.player_state.get(f"{key_a}_HAS_ROAD", False)
    has_road_b = st.player_state.get(f"{key_b}_HAS_ROAD", False)
    
    # Categorize cause
    if vp_a >= 9:
        cause = "CLOSE_FINISH_DICE_LUCK"
    elif open_b["total_pips"] - open_a["total_pips"] >= 3.0 or open_a["variety"] <= 2:
        cause = "OPENING_PIPS_DEFICIT"
    elif has_road_b and not has_road_a and (road_len_b - road_len_a >= 3):
        cause = "OUT_EXPANDED_LONGEST_ROAD"
    elif has_army_b and not has_army_a and army_b >= 3:
        cause = "LOST_LARGEST_ARMY"
    elif cities_b > cities_a:
        cause = "OUT_CITIED"
    elif vp_a >= 8:
        cause = "COMPETITIVE_FINISH"
    else:
        cause = "GENERAL_PRODUCTION_DEFICIT"
        
    return {
        "cause": cause,
        "vp_a": vp_a,
        "vp_b": vp_b,
        "pub_vp_a": pub_vp_a,
        "pub_vp_b": pub_vp_b,
        "open_a": open_a,
        "open_b": open_b,
        "settlements_a": settlements_a,
        "cities_a": cities_a,
        "settlements_b": settlements_b,
        "cities_b": cities_b,
        "army_a": army_a,
        "army_b": army_b,
        "has_army_a": has_army_a,
        "has_army_b": has_army_b,
        "road_len_a": road_len_a,
        "road_len_b": road_len_b,
        "has_road_a": has_road_a,
        "has_road_b": has_road_b,
        "turns": st.num_turns,
    }


def replay_game_and_diagnose(path_a, path_b, board_seed, gi, key=DEFAULT_KEY, vps=10):
    swap = gi % 2 == 1
    color_a = COLORS[1] if swap else COLORS[0]
    color_b = COLORS[0] if swap else COLORS[1]
    
    a_proc = AgentProcess(path_a)
    b_proc = AgentProcess(path_b)
    seat_a = Seat(color_a, a_proc, clocked=True)
    seat_b = Seat(color_b, b_proc, clocked=True)
    order = [seat_b, seat_a] if swap else [seat_a, seat_b]
    
    try:
        random.seed(board_seed)
        game = Game(order, seed=board_seed, vps_to_win=vps)
        random.seed(dice_seed(board_seed, gi // 2, key))
        s = (board_seed * 1_000_003 + gi) & 0xFFFFFFFF
        seat_a.begin_game(s)
        seat_b.begin_game(s ^ 0xABCDEF)
        winner = game.play()
        seat_a.end_game()
        seat_b.end_game()
        
        is_loss = (winner == color_b)
        diag = None
        if is_loss:
            diag = analyze_loss(game, color_a, color_b)
            diag["board_seed"] = board_seed
            diag["game_index"] = gi
            diag["a_first"] = (not swap)
        return winner, is_loss, diag
    finally:
        a_proc.stop()
        b_proc.stop()


def diagnose_chunk(path_a, path_b, games=30, seed0=990000):
    print(f"=== Accurate loss diagnostics: {path_a} vs {path_b} over {games} games ===")
    losses = []
    wins = 0
    seeds = [seed0 + i // 2 for i in range(games)]
    for gi in range(games):
        b_seed = seeds[gi]
        winner, is_loss, diag = replay_game_and_diagnose(path_a, path_b, b_seed, gi)
        if not is_loss:
            wins += 1
        else:
            losses.append(diag)
        status = "WIN" if not is_loss else f"LOSS ({diag['cause']}, VP {diag['vp_a']}-{diag['vp_b']})"
        print(f"Game {gi+1}/{games}: {status}")

    print("\n" + "="*50)
    print(f"Summary: {wins}/{games} wins ({wins/games*100:.1f}%), {len(losses)} losses")
    print("="*50)
    
    if not losses:
        print("No losses to diagnose!")
        return
        
    causes = {}
    for l in losses:
        causes[l["cause"]] = causes.get(l["cause"], 0) + 1
        
    print("\nLoss Breakdown by Cause:")
    for c, count in sorted(causes.items(), key=lambda x: -x[1]):
        pct = count / len(losses) * 100
        print(f"  {c:<30} {count:>3} ({pct:>5.1f}%)")
        
    avg_vp_a = sum(l["vp_a"] for l in losses) / len(losses)
    avg_open_pips_a = sum(l["open_a"]["total_pips"] for l in losses) / len(losses)
    avg_open_pips_b = sum(l["open_b"]["total_pips"] for l in losses) / len(losses)
    avg_open_var_a = sum(l["open_a"]["variety"] for l in losses) / len(losses)
    avg_open_var_b = sum(l["open_b"]["variety"] for l in losses) / len(losses)
    avg_settle_a = sum(l["settlements_a"] for l in losses) / len(losses)
    avg_settle_b = sum(l["settlements_b"] for l in losses) / len(losses)
    avg_city_a = sum(l["cities_a"] for l in losses) / len(losses)
    avg_city_b = sum(l["cities_b"] for l in losses) / len(losses)
    lost_army = sum(1 for l in losses if l["has_army_b"])
    lost_road = sum(1 for l in losses if l["has_road_b"])
    
    print("\nKey Metrics in Losses:")
    print(f"  Avg final VPs (A vs B):             {avg_vp_a:.1f} vs 10.0")
    print(f"  Avg OPENING pips (A vs B):          {avg_open_pips_a:.1f} vs {avg_open_pips_b:.1f}")
    print(f"  Avg OPENING resource variety (A vs B): {avg_open_var_a:.1f} vs {avg_open_var_b:.1f}")
    print(f"  Avg settlements built (A vs B):     {avg_settle_a:.1f} vs {avg_settle_b:.1f}")
    print(f"  Avg cities built (A vs B):          {avg_city_a:.1f} vs {avg_city_b:.1f}")
    print(f"  B secured Largest Army in:          {lost_army}/{len(losses)} ({lost_army/len(losses)*100:.1f}%)")
    print(f"  B secured Longest Road in:          {lost_road}/{len(losses)} ({lost_road/len(losses)*100:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="agent/agent.py")
    p.add_argument("--b", default="examples/f_agent.py")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--seed0", type=int, default=990000)
    args = p.parse_args()
    diagnose_chunk(args.a, args.b, args.games, args.seed0)
