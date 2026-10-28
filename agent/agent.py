"""Master Catan Agent v10: Championship Engine.
- Game-Theoretic Opening (S1 Anchor + Competitive Pair S2/S3 + Hand-Optimized Order + d=2 Road Routing)
- Pre-Roll Knight Trigger (Unblock prime hexes before rolling)
- Strategic Robber (Avoid self-touch, prioritize opp cities and Ore/Wheat bottlenecks)
- Unlocked City Macro (Direct trade-to-city recognition)
- Exact 10 VP Finisher (DFS winning sequences)
"""
import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from catanatron.models.board import get_node_distances, STATIC_GRAPH
from catanatron.models.enums import ActionType, SETTLEMENT, CITY, RESOURCES
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS

import evalfn

F_FN = base_fn(DEFAULT_WEIGHTS)
VALUE_FN = evalfn.value

PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
DISTANCES = get_node_distances()


def get_node_stats(cmap, node):
    adj_tiles = cmap.adjacent_tiles.get(node, [])
    land_tiles = [t for t in adj_tiles if getattr(t, "resource", None) is not None]
    res_pips = {r: 0.0 for r in RESOURCES}
    numbers = set()
    total_pips = 0.0
    for t in land_tiles:
        p = PIPS.get(t.number, 0)
        total_pips += p
        res_pips[t.resource] += p
        if t.number is not None:
            numbers.add(t.number)

    ports = []
    port_nodes = getattr(cmap, "port_nodes", {})
    for port_res, pnodes in port_nodes.items():
        if node in pnodes:
            ports.append(port_res)

    return {
        "node": node,
        "land_count": len(land_tiles),
        "total_pips": total_pips,
        "res_pips": res_pips,
        "numbers": numbers,
        "ports": ports,
    }


def score_joint_portfolio(stats1, stats2, cmap=None):
    res_pips = {r: stats1["res_pips"][r] + stats2["res_pips"][r] for r in RESOURCES}
    numbers = stats1["numbers"] | stats2["numbers"]
    ports = stats1["ports"] + stats2["ports"]

    # Diminishing returns on excess sheep without a port
    has_sheep_port = ("SHEEP" in ports)
    sheep_prod = res_pips["SHEEP"]
    if not has_sheep_port and sheep_prod > 5.0:
        effective_sheep = 5.0 + (sheep_prod - 5.0) * 0.2
    else:
        effective_sheep = sheep_prod

    effective_total = (res_pips["WOOD"] + res_pips["BRICK"] +
                       res_pips["WHEAT"] + res_pips["ORE"] + effective_sheep)

    score = effective_total * 10.0

    # 3-hex junctions preference
    score += (stats1["land_count"] + stats2["land_count"]) * 6.0

    # Number spread (roll diversity across turns prevents card droughts)
    score += len(numbers) * 7.0

    # Variety bonus (calibrated: 35 points = 3.5 pips)
    variety = sum(1 for p in res_pips.values() if p > 0.01)
    if variety == 5:
        score += 35.0
    elif variety == 4 and res_pips["SHEEP"] < 0.01:
        score += 35.0

    # Resource requirements
    # 1. Ore: essential for cities
    if res_pips["ORE"] < 0.01:
        has_ore_port = any(p is not None and res_pips.get(p, 0.0) >= 6.0 for p in ports)
        score -= (40.0 if has_ore_port else 100.0)
    elif res_pips["ORE"] < 2.5:
        score -= 25.0

    # 2. Wheat: essential for settlements and cities
    if res_pips["WHEAT"] < 0.01:
        has_wheat_port = any(p is not None and res_pips.get(p, 0.0) >= 6.0 for p in ports)
        score -= (40.0 if has_wheat_port else 100.0)
    elif res_pips["WHEAT"] < 2.5:
        score -= 25.0

    # 3. Wood and Brick: essential for road expansion
    has_road = (res_pips["WOOD"] > 0.01 and res_pips["BRICK"] > 0.01)
    if not has_road:
        has_wb_port = any(p is not None and res_pips.get(p, 0.0) >= 6.0 for p in ports)
        score -= (30.0 if has_wb_port else 80.0)
    else:
        score += 25.0
        score += min(res_pips["WOOD"], res_pips["BRICK"]) * 3.0
        score -= abs(res_pips["WOOD"] - res_pips["BRICK"]) * 1.5

    # 4. City synergy (Ore + Wheat)
    if res_pips["ORE"] >= 2.5 and res_pips["WHEAT"] >= 2.5:
        score += 25.0
        score += min(res_pips["ORE"], res_pips["WHEAT"] * 1.5) * 4.0

    # 5. Port synergies (reward matching port only when production is strong)
    for p in ports:
        if p is None:
            if effective_total >= 20:
                score += 15.0
        else:
            prod = res_pips.get(p, 0.0)
            if prod >= 8.0:
                score += 45.0
            elif prod >= 6.0:
                score += 30.0
            elif prod >= 4.0:
                score += 15.0

    return score


def node_start_hand_value(stats):
    """Evaluate quality of starting hand if this node is placed second (giving initial cards)."""
    rp = stats["res_pips"]
    val = 0.0
    if rp["WOOD"] > 0.01:
        val += 25.0
    if rp["BRICK"] > 0.01:
        val += 25.0
    if rp["WHEAT"] > 0.01:
        val += 15.0
    if rp["SHEEP"] > 0.01:
        val += 8.0
    if rp["ORE"] > 0.01:
        val += 5.0
    return val


class Agent:
    def __init__(self):
        self.rng = random.Random(20261017)
        self.planned_opening_s3 = None
        self._warmup()

    def _warmup(self):
        try:
            from catan_kit.obs import masked_copy
            from catanatron import Game
            from catanatron.models.player import SimplePlayer, Color

            for seed in (123, 456):
                g = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], seed=seed)
                for _ in range(14):
                    if g.winning_color() is not None or not g.playable_actions:
                        break
                    g.execute(g.playable_actions[0])
                obs = _FakeObs(g, Color.RED)
                self.decide(obs, list(g.playable_actions))
            gc.collect()
            gc.freeze()
        except Exception:
            pass

    def decide(self, obs, playable_actions) -> int:
        try:
            return self._decide(obs, playable_actions)
        except Exception:
            try:
                return self._fallback(obs, playable_actions)
            except Exception:
                return 0

    def _decide(self, obs, playable_actions):
        types = {a.action_type.name for a in playable_actions}

        # 1. Opening settlement placement: Master Joint Portfolio Engine
        if "BUILD_SETTLEMENT" in types and len(playable_actions) > 10:
            return self._opening_settlement(obs, playable_actions)

        # 2. Opening road placement: Point towards best expansion node at distance 2
        if "BUILD_ROAD" in types and getattr(obs, "turn", 1) <= 4 and len(playable_actions) <= 4:
            return self._opening_road(obs, playable_actions)

        # 3. Exact 10 VP Finisher (check whenever actual VPs >= 8 and clock has margin)
        if getattr(obs, "my_actual_vps", 0) >= 8 and (obs.cpu_left is None or obs.cpu_left > 0.15):
            world0 = obs.sample_world(self.rng)
            w = self._win_now(world0, obs.me, playable_actions)
            if w is not None:
                return w

        # 4. Pre-Roll Knight trigger: unblock our high-production hexes BEFORE rolling
        if "PLAY_KNIGHT_CARD" in types and "ROLL" in types:
            robber_coord = obs.board.robber_coordinate
            tile = obs.board.map.tiles.get(robber_coord)
            if tile is not None and getattr(tile, "number", None) in (5, 6, 8, 9, 4, 10):
                my_nodes = {n for n, v in obs.board.buildings.items() if v is not None and v[0] == obs.me}
                if set(tile.nodes.values()) & my_nodes:
                    for i, a in enumerate(playable_actions):
                        if a.action_type == ActionType.PLAY_KNIGHT_CARD:
                            return i

        # 5. Robber placement optimization: Never rob self, prioritize opp cities and bottlenecks
        if "MOVE_ROBBER" in types:
            return self._robber_move(obs, playable_actions)

        # 6. Master Lookahead: Unlocked Macro for City, Settlement, and Road-Takeover
        world = obs.sample_world(self.rng)
        has_city_already = any(a.action_type == ActionType.BUILD_CITY for a in playable_actions)
        has_settlement_already = any(a.action_type == ActionType.BUILD_SETTLEMENT for a in playable_actions)
        my_key = f"P{obs.board.color_to_index[obs.me]}" if hasattr(obs.board, "color_to_index") else None

        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            succ = world.copy()
            try:
                succ.execute(action)
            except Exception:
                continue
            v = VALUE_FN(succ, obs.me)

            # Unlocked Macro: evaluate if a trade immediately enables a high-impact build!
            if action.action_type == ActionType.MARITIME_TRADE and succ.state.current_color() == obs.me:
                trade_giving = action.value[0] if isinstance(action.value, tuple) else None

                # 1. City Unlock: 1-step or 2-step (port sequence)
                if not has_city_already:
                    for a2 in succ.playable_actions:
                        if a2.action_type == ActionType.BUILD_CITY:
                            s2 = succ.copy()
                            try:
                                s2.execute(a2)
                                v2 = VALUE_FN(s2, obs.me)
                                if v2 > v:
                                    v = v2
                            except Exception:
                                pass
                        elif a2.action_type == ActionType.MARITIME_TRADE:
                            # 2-step port trade towards city
                            s2 = succ.copy()
                            try:
                                s2.execute(a2)
                                for a3 in s2.playable_actions:
                                    if a3.action_type == ActionType.BUILD_CITY:
                                        s3 = s2.copy()
                                        s3.execute(a3)
                                        v3 = VALUE_FN(s3, obs.me)
                                        if v3 > v:
                                            v = v3
                            except Exception:
                                pass

                # 2. Settlement Unlock: NEVER trade away Ore (reserve for cities)
                if not has_settlement_already and trade_giving != "ORE":
                    for a2 in succ.playable_actions:
                        if a2.action_type == ActionType.BUILD_SETTLEMENT:
                            s2 = succ.copy()
                            try:
                                s2.execute(a2)
                                v2 = VALUE_FN(s2, obs.me)
                                if v2 > v:
                                    v = v2
                            except Exception:
                                pass

                # 3. Longest Road Takeover: if a trade enables a road that captures Longest Road (+2 VP)
                for a2 in succ.playable_actions:
                    if a2.action_type == ActionType.BUILD_ROAD:
                        s2 = succ.copy()
                        try:
                            s2.execute(a2)
                            if my_key and s2.state.player_state.get(f"{my_key}_HAS_ROAD", False):
                                v2 = VALUE_FN(s2, obs.me)
                                if v2 > v:
                                    v = v2
                        except Exception:
                            pass

            if best_v is None or v > best_v:
                best_i, best_v = i, v

        return best_i

    def _opening_settlement(self, obs, playable_actions):
        cmap = obs.board.map
        settlement_idxs = [i for i, a in enumerate(playable_actions)
                           if a.action_type == ActionType.BUILD_SETTLEMENT]
        if not settlement_idxs:
            return 0

        all_buildings = obs.board.buildings
        my_existing = [n for n, v in all_buildings.items() if v is not None and v[0] == obs.me]
        opp_existing = [n for n, v in all_buildings.items() if v is not None and v[0] != obs.me]

        legal_action_nodes = {playable_actions[i].value: i for i in settlement_idxs}
        stats = {n: get_node_stats(cmap, n) for n in legal_action_nodes}

        # Reset plan when at root
        if len(opp_existing) == 0:
            self.planned_opening_s3 = None

        # CASE 1: Second seat placing S2 (opp has 1 settlement, we have 0)
        # We place S2 AND S3 back to back! Competitive joint pair optimization!
        if len(my_existing) == 0 and len(opp_existing) == 1:
            cand_nodes = [n for n in legal_action_nodes if stats[n]["land_count"] >= 2]
            opp_s1 = opp_existing[0]
            opp_s1_stats = get_node_stats(cmap, opp_s1)
            blocked_by_opp = {opp_s1} | {x for x, d in DISTANCES[opp_s1].items() if d <= 1}

            best_pair = None
            best_net = -float("inf")
            for i, n1 in enumerate(cand_nodes):
                for n2 in cand_nodes[i + 1:]:
                    if DISTANCES[n1][n2] >= 2:
                        p1_sc = score_joint_portfolio(stats[n1], stats[n2], cmap)

                        # Mild tie-breaker for opponent denial (do not sacrifice own pips!)
                        blocked_s4 = blocked_by_opp | {n1, n2} | {x for x, d in DISTANCES[n1].items() if d <= 1} | {x for x, d in DISTANCES[n2].items() if d <= 1}
                        avail_s4 = [x for x in cmap.land_nodes if x not in blocked_s4 and stats.get(x, get_node_stats(cmap, x))["land_count"] >= 2]

                        p0_best_s4_sc = max((score_joint_portfolio(opp_s1_stats, stats.get(s4, get_node_stats(cmap, s4)), cmap)
                                             for s4 in avail_s4), default=0.0)

                        net = p1_sc - 0.08 * p0_best_s4_sc
                        if net > best_net:
                            best_net = net
                            best_pair = (n1, n2)

            if best_pair is not None:
                h1 = node_start_hand_value(stats[best_pair[0]])
                h2 = node_start_hand_value(stats[best_pair[1]])
                if h1 >= h2:
                    first_node, second_node = best_pair[1], best_pair[0]
                else:
                    first_node, second_node = best_pair[0], best_pair[1]

                self.planned_opening_s3 = second_node
                return legal_action_nodes[first_node]

        # CASE 2: Second seat placing S3 (we already have S2)
        if len(my_existing) == 1 and self.planned_opening_s3 in legal_action_nodes:
            choice = self.planned_opening_s3
            self.planned_opening_s3 = None
            return legal_action_nodes[choice]

        # CASE 3: We have an existing settlement (Seat 0 placing S4 or Seat 1 fallback)
        if len(my_existing) >= 1:
            s1_node = my_existing[0]
            s1_stats = get_node_stats(cmap, s1_node)
            best_node = None
            best_sc = -float("inf")
            for n, idx in legal_action_nodes.items():
                if stats[n]["land_count"] < 2:
                    continue
                sc = score_joint_portfolio(s1_stats, stats[n], cmap)
                if sc > best_sc:
                    best_sc = sc
                    best_node = n
            if best_node is not None:
                return legal_action_nodes[best_node]

        # CASE 4: First seat placing S1 (board is completely empty)
        # Grab the undisputed highest-quality 3-hex junction on the board!
        cand_nodes = [n for n in legal_action_nodes if stats[n]["land_count"] == 3]
        if not cand_nodes:
            cand_nodes = list(legal_action_nodes.keys())

        best_s1 = max(cand_nodes, key=lambda n: (
            stats[n]["total_pips"] * 10.0 +
            stats[n]["land_count"] * 6.0 +
            len(stats[n]["numbers"]) * 6.0 +
            (stats[n]["res_pips"]["ORE"] + stats[n]["res_pips"]["WHEAT"]) * 3.5 +
            min(stats[n]["res_pips"]["WOOD"], stats[n]["res_pips"]["BRICK"]) * 3.0
        ))
        return legal_action_nodes[best_s1]

    def _opening_road(self, obs, playable_actions):
        """Point opening road towards the legal distance-2 expansion node with highest pips."""
        cmap = obs.board.map
        road_idxs = [i for i, a in enumerate(playable_actions)
                     if a.action_type == ActionType.BUILD_ROAD]
        if not road_idxs:
            return 0

        all_buildings = obs.board.buildings
        existing_nodes = [n for n, v in all_buildings.items() if v is not None]
        my_nodes = [n for n, v in all_buildings.items() if v is not None and v[0] == obs.me]

        best_i = road_idxs[0]
        best_score = -float("inf")

        for i in road_idxs:
            edge = playable_actions[i].value
            origin = edge[0] if edge[0] in my_nodes else edge[1]
            neighbor = edge[1] if origin == edge[0] else edge[0]

            d2_nodes = [n2 for n2 in STATIC_GRAPH.neighbors(neighbor) if n2 != origin]
            legal_d2 = [n2 for n2 in d2_nodes
                        if all(DISTANCES[n2][b] >= 2 for b in existing_nodes)]

            if legal_d2:
                best_target_score = 0.0
                for n2 in legal_d2:
                    st = get_node_stats(cmap, n2)
                    t_score = st["total_pips"] * 10.0 + st["land_count"] * 5.0
                    if st["ports"]:
                        t_score += 15.0
                    if t_score > best_target_score:
                        best_target_score = t_score
                score = best_target_score
            else:
                st_neigh = get_node_stats(cmap, neighbor)
                score = st_neigh["total_pips"] * 2.0 - 50.0

            if score > best_score:
                best_score = score
                best_i = i

        return best_i

    def _robber_move(self, obs, playable_actions):
        """Championship Robber: Never harm self, prioritize opp cities and Ore/Wheat bottlenecks."""
        cmap = obs.board.map
        my_nodes = {n for n, v in obs.board.buildings.items() if v is not None and v[0] == obs.me}
        opp_nodes = {n: v[1] for n, v in obs.board.buildings.items() if v is not None and v[0] != obs.me}

        best_i = 0
        best_score = -float("inf")

        for i, a in enumerate(playable_actions):
            if a.action_type != ActionType.MOVE_ROBBER:
                continue
            coord, target_color = a.value
            tile = cmap.tiles.get(coord)
            if tile is None:
                continue

            tile_nodes = set(tile.nodes.values())
            self_touch = tile_nodes & my_nodes
            if self_touch:
                score = -2000.0 - len(self_touch) * 1000.0
            else:
                pips = PIPS.get(tile.number, 0)
                score = pips * 12.0

                # Opponent building bonus (Cities are double production = double damage!)
                opp_touch = tile_nodes & set(opp_nodes.keys())
                for on in opp_touch:
                    if opp_nodes[on] == CITY:
                        score += 50.0
                    else:
                        score += 20.0

                # Resource priority: blocking ORE or WHEAT hurts cities and settlements!
                res = getattr(tile, "resource", None)
                if res in ("ORE", "WHEAT"):
                    score += 30.0
                elif res in ("WOOD", "BRICK"):
                    score += 15.0

                # Steal bonus
                if target_color is not None and target_color != obs.me:
                    score += 25.0

            if score > best_score:
                best_score = score
                best_i = i

        return best_i

    def _win_now(self, world, me, playable_actions):
        t0 = time.process_time()

        def dfs(game, depth):
            if time.process_time() - t0 > 0.012:
                return None
            if game.winning_color() == me:
                return []
            if depth == 0:
                return None
            for a in game.playable_actions:
                if a.action_type in (ActionType.ROLL, ActionType.END_TURN):
                    continue
                s = game.copy()
                try:
                    s.execute(a)
                except Exception:
                    continue
                res = dfs(s, depth - 1)
                if res is not None:
                    return [a] + res
            return None

        try:
            seq = dfs(world, depth=3)
            if seq:
                first = seq[0]
                for i, a in enumerate(playable_actions):
                    if a.action_type == first.action_type and a.value == first.value:
                        return i
        except Exception:
            pass
        return None

    def _fallback(self, obs, playable_actions):
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            succ = world.copy()
            try:
                succ.execute(action)
            except Exception:
                continue
            v = F_FN(succ, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i


class _FakeObs:
    def __init__(self, game, me):
        self.me = me
        self.cpu_left = 1.0
        self.my_actual_vps = 0
        self.turn = 1
        self.board = game.state.board
        self._game = game

    def sample_world(self, rng):
        from catan_kit.obs import masked_copy
        return masked_copy(self._game, self.me, rng)
