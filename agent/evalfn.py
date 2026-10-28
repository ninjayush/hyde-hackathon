"""Evaluation function: F's 13 terms plus calibrated race, city, port, and discard terms.
"""
import json
import os

from catanatron.features import get_node_production
from catanatron.models.enums import SETTLEMENT, CITY, RESOURCES
from catanatron.players.value import base_fn
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

W = {
    "public_vps": 3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    "reachable_production_0": 0,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10,
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
    # --- extra calibrated terms ---
    "city_value": 5e7,
    "port_value": 1.5e7,
    "port_opp": -1e6,
    "discard_risk": -5e6,
    "opp_dev_vp": -1e6,
    "army_progress": 1.5e7,
    "road_progress": 1.5e7,
    "city_plan": 1000.0,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_HERE, "weights.json")
if os.path.exists(_path):
    with open(_path) as f:
        W.update(json.load(f))

_F_KEYS = ("public_vps", "production", "enemy_production", "num_tiles",
           "reachable_production_0", "reachable_production_1", "buildable_nodes",
           "longest_road", "hand_synergy", "hand_resources", "discard_penalty",
           "hand_devs", "army_size")
_F_FN = base_fn({k: W[k] for k in _F_KEYS})

_VP_SHARE = 5 / 25


def value(game, p0_color):
    st = game.state
    board = st.board
    j = 1 - st.color_to_index[p0_color]
    opp = st.colors[j]
    key = player_key(st, p0_color)
    opp_key = player_key(st, opp)

    v = _F_FN(game, p0_color)

    # 1. Ports: only valuable if we actually produce that resource!
    ports = board.get_player_port_resources(p0_color)
    ps = 0.0
    owned_nodes = st.buildings_by_color[p0_color][SETTLEMENT] + st.buildings_by_color[p0_color][CITY]
    for r in RESOURCES:
        if r in ports:
            prod_r = sum(get_node_production(board.map, n, r) for n in owned_nodes)
            ps += W["port_value"] * min(prod_r / (5 / 36), 2.0)
    if None in ports:
        total_prod = sum(get_node_production(board.map, n, r) for n in owned_nodes for r in RESOURCES)
        ps += W["port_value"] * 0.3 * min(total_prod / (15 / 36), 1.5)

    opp_ports = board.get_player_port_resources(opp)
    ps_opp = 0.0
    opp_nodes = st.buildings_by_color[opp][SETTLEMENT] + st.buildings_by_color[opp][CITY]
    for r in RESOURCES:
        if r in opp_ports:
            prod_r_opp = sum(get_node_production(board.map, n, r) for n in opp_nodes)
            ps_opp += W["port_value"] * min(prod_r_opp / (5 / 36), 2.0)
    if None in opp_ports:
        total_prod_opp = sum(get_node_production(board.map, n, r) for n in opp_nodes for r in RESOURCES)
        ps_opp += W["port_value"] * 0.3 * min(total_prod_opp / (15 / 36), 1.5)
    v += ps + W["port_opp"] * ps_opp / max(W["port_value"], 1)

    # 2. Opponent hidden VP: expected VP cards in unplayed devs
    n_devs = player_num_dev_cards(st, opp)
    v += W["opp_dev_vp"] * n_devs * _VP_SHARE

    # 3. Longest-road race (capped to prevent endless useless roads!)
    has_road = st.player_state.get(f"{key}_HAS_ROAD", False)
    opp_has_road = st.player_state.get(f"{opp_key}_HAS_ROAD", False)
    my_lr = get_longest_road_length(st, p0_color)
    opp_lr = get_longest_road_length(st, opp)

    if opp_has_road:
        # Opponent holds Longest Road: reward catching up and taking it (+2 to us, -2 to opp)
        v += W["road_progress"] * min(my_lr - opp_lr, 1)
    elif has_road:
        # We hold Longest Road: defend margin up to +2, capped so extra roads are NOT rewarded
        v += W["road_progress"] * min(my_lr - opp_lr, 2)
    else:
        # Unclaimed: race to 5
        if my_lr >= 3 or opp_lr >= 3:
            v += W["road_progress"] * (min(my_lr, 5) - min(opp_lr, 5))

    # 4. Largest-army race (capped)
    has_army = st.player_state.get(f"{key}_HAS_ARMY", False)
    opp_has_army = st.player_state.get(f"{opp_key}_HAS_ARMY", False)
    my_k = get_played_dev_cards(st, p0_color, "KNIGHT")
    opp_k = get_played_dev_cards(st, opp, "KNIGHT")

    if opp_has_army:
        v += W["army_progress"] * min(my_k - opp_k, 1)
    elif has_army:
        v += W["army_progress"] * min(my_k - opp_k, 2)
    else:
        if my_k >= 2 or opp_k >= 2:
            v += W["army_progress"] * (min(my_k, 3) - min(opp_k, 3))

    # 5. City bonus: reward converting settlements to cities (vital double production + 1 VP)
    num_cities = len(st.buildings_by_color[p0_color][CITY])
    v += W.get("city_value", 5e7) * num_cities

    # 6. Discard risk
    n_hand = player_num_resource_cards(st, p0_color)
    if n_hand > 7:
        v += W["discard_risk"] * (n_hand - 7)

    return v
