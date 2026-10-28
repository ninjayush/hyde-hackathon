"""Experimental Agent v3: Master Opening Engine (d=2 Road Routing + Hand Optimization) + Smart Robber + Unlocked Build Macro.
"""
import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from catanatron.models.board import get_node_distances, STATIC_GRAPH
from catanatron.models.enums import ActionType, SETTLEMENT, CITY, ROAD, RESOURCES
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

    # Diminishing returns on excess sheep without a matching port
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
    score += (stats1["land_count"] + stats2["land_count"]) * 5.0

    # Number spread (roll diversity across turns prevents card droughts)
    score += len(numbers) * 7.0

    # Hard resource requirements in 2-player Catan
    # 1. Ore requirement: strictly necessary for cities
    if res_pips["ORE"] < 0.01:
        score -= 250.0  # Fatal: cannot build cities
    elif res_pips["ORE"] < 3.0:
        score -= 50.0

    # 2. Wheat requirement: strictly necessary for cities and settlements
    if res_pips["WHEAT"] < 0.01:
        score -= 250.0  # Fatal: cannot build anything
    elif res_pips["WHEAT"] < 3.0:
        score -= 40.0

    # 3. Wood and Brick requirements: strictly necessary for road expansion
    has_road = (res_pips["WOOD"] > 0.01 and res_pips["BRICK"] > 0.01)
    if not has_road:
        has_strong_port = any(p is not None and res_pips.get(p, 0.0) >= 6.0 for p in ports)
        if not has_strong_port:
            score -= 200.0  # Fatal: cannot build roads
        else:
            score -= 30.0
    else:
        score += 30.0
        # Road balance bonus
        score -= abs(res_pips["WOOD"] - res_pips["BRICK"]) * 1.5

    # 4. Variety bonus
    variety = sum(1 for p in res_pips.values() if p > 0.01)
    if variety == 5:
        score += 70.0
    elif variety == 4 and res_pips["SHEEP"] < 0.01:
        score += 40.0

    # 5. Ore + Wheat City synergy (the core win condition)
    if res_pips["ORE"] >= 3.0 and res_pips["WHEAT"] >= 3.0:
        score += 35.0
        score += min(res_pips["ORE"], res_pips["WHEAT"] * 1.5) * 6.0

    score += res_pips["ORE"] * 2.0 + res_pips["WHEAT"] * 2.0

    # 6. Port synergies (reward matching port only when production is strong)
    for p in ports:
        if p is None:
            if effective_total >= 20:
                score += 15.0
        else:
            prod = res_pips.get(p, 0.0)
            if prod >= 8.0:
                score += 50.0
            elif prod >= 6.0:
                score += 35.0
            elif prod >= 4.0:
                score += 15.0

    return score


def node_start_hand_value(stats):
    """Evaluate quality of starting hand if this node is placed second (giving initial cards)."""
    rp = stats["res_pips"]
    val = 0.0
    if rp["WOOD"] > 0.01:
        val += 20.0
    if rp["BRICK"] > 0.01:
        val += 20.0
    if rp["WHEAT"] > 0.01:
        val += 12.0
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

        # 4. Robber placement optimization: Never rob self, prioritize opp cities / bottlenecks
        if "MOVE_ROBBER" in types:
            return self._robber_move(obs, playable_actions)

        # 5. Fast 1-ply lookahead with Unlocked Build Macro
        world = obs.sample_world(self.rng)

        pre_builds = {a.action_type for a in playable_actions
                      if a.action_type in (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD)}

        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            succ = world.copy()
            try:
                succ.execute(action)
            except Exception:
                continue
            v = VALUE_FN(succ, obs.me)

            # Unlocked build macro: if a trade directly enables an unplayable build, evaluate post-build
            if action.action_type == ActionType.MARITIME_TRADE and succ.state.current_color() == obs.me:
                for a2 in succ.playable_actions:
                    if a2.action_type in (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
                        if a2.action_type not in pre_builds:
                            s2 = succ.copy()
                            try:
                                s2.execute(a2)
                                v2 = VALUE_FN(s2, obs.me)
                                if v2 > v:
                                    v = v2
                            except Exception:
                                continue

            if best_v is None or v > best_v:
                best_i, best_v = i, v

        return best_i

    def _opening_settlement(self, obs, playable_actions):
        cmap = obs.board.map
        settlement_idxs = [i for i, a in enumerate(playable_actions)
                           if a.action_type == ActionType.BUILD_SETTLEMENT]
        if not settlement_idxs:
            return 0

        # Current buildings on the board
        all_buildings = obs.board.buildings
        my_existing = [n for n, v in all_buildings.items() if v is not None and v[0] == obs.me]
        opp_existing = [n for n, v in all_buildings.items() if v is not None and v[0] != obs.me]

        legal_action_nodes = {playable_actions[i].value: i for i in settlement_idxs}
        stats = {n: get_node_stats(cmap, n) for n in legal_action_nodes}

        # CASE 1: Second seat placing S2 (opp has 1 settlement, we have 0)
        # We place S2 AND S3 back to back! Joint pair optimization across all legal pairs!
        if len(my_existing) == 0 and len(opp_existing) == 1:
            cand_nodes = [n for n in legal_action_nodes if stats[n]["land_count"] >= 2]
            best_pair = None
            best_sc = -float("inf")
            for i, n1 in enumerate(cand_nodes):
                for n2 in cand_nodes[i + 1:]:
                    if DISTANCES[n1][n2] >= 2:
                        sc = score_joint_portfolio(stats[n1], stats[n2], cmap)
                        if sc > best_sc:
                            best_sc = sc
                            best_pair = (n1, n2)

            if best_pair is not None:
                # Order the pair so the one with Wood+Brick is placed SECOND (giving starting cards!)
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
        # Minimax lookahead: pick S1 that guarantees the best complementary pair
        cand_nodes = [n for n in legal_action_nodes
                      if stats[n]["land_count"] == 3 and stats[n]["total_pips"] >= 9]
        if not cand_nodes:
            cand_nodes = list(legal_action_nodes.keys())

        best_s1 = cand_nodes[0]
        best_net = -float("inf")

        for n1 in cand_nodes:
            blocked_by_s1 = {n1} | {x for x, d in DISTANCES[n1].items() if d <= 1}
            avail_opp = [x for x in legal_action_nodes if x not in blocked_by_s1 and stats[x]["land_count"] >= 2]

            # Opponent's best pair response
            opp_best_sc = 0.0
            opp_best_pair = None
            for i, a in enumerate(avail_opp):
                for b in avail_opp[i + 1:]:
                    if DISTANCES[a][b] >= 2:
                        sc = score_joint_portfolio(stats[a], stats[b], cmap)
                        if sc > opp_best_sc:
                            opp_best_sc = sc
                            opp_best_pair = (a, b)

            # S4 completion from remaining nodes
            blocked_by_opp = set()
            if opp_best_pair:
                for on in opp_best_pair:
                    blocked_by_opp.update({on} | {x for x, d in DISTANCES[on].items() if d <= 1})

            avail_s4 = [x for x in legal_action_nodes
                        if x not in blocked_by_s1 and x not in blocked_by_opp and stats[x]["land_count"] >= 2]

            my_best_pair_sc = -float("inf")
            for s4 in avail_s4:
                sc = score_joint_portfolio(stats[n1], stats[s4], cmap)
                if sc > my_best_pair_sc:
                    my_best_pair_sc = sc

            net_sc = my_best_pair_sc - opp_best_sc * 0.3
            if net_sc > best_net:
                best_net = net_sc
                best_s1 = n1

        return legal_action_nodes.get(best_s1, settlement_idxs[0])

    def _opening_road(self, obs, playable_actions):
        """Point opening road towards the legal distance-2 expansion node with highest pips."""
        cmap = obs.board.map
        road_idxs = [i for i, a in enumerate(playable_actions)
                     if a.action_type == ActionType.BUILD_ROAD]
        if not road_idxs:
            return 0

        all_buildings = obs.board.buildings
        existing_nodes = [n for n, v in all_buildings.items() if v is not None]

        # Determine which settlement was just built
        my_nodes = [n for n, v in all_buildings.items() if v is not None and v[0] == obs.me]
        
        # Check edges
        best_i = road_idxs[0]
        best_score = -float("inf")

        for i in road_idxs:
            edge = playable_actions[i].value
            # Identify origin (settlement node) and neighbor (distance 1 node)
            origin = edge[0] if edge[0] in my_nodes else edge[1]
            neighbor = edge[1] if origin == edge[0] else edge[0]

            # Look for distance-2 nodes reachable through neighbor
            d2_nodes = [n2 for n2 in STATIC_GRAPH.neighbors(neighbor) if n2 != origin]
            # Must be legal to settle (distance >= 2 from all existing buildings)
            legal_d2 = [n2 for n2 in d2_nodes
                        if all(DISTANCES[n2][b] >= 2 for b in existing_nodes)]

            if legal_d2:
                # Find best expansion target
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
                # Dead end or coast: evaluate neighbor weakly
                st_neigh = get_node_stats(cmap, neighbor)
                score = st_neigh["total_pips"] * 2.0 - 50.0

            if score > best_score:
                best_score = score
                best_i = i

        return best_i

    def _robber_move(self, obs, playable_actions):
        """Smart robber: strictly avoid self-damage, prioritize opp cities and bottleneck tiles."""
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
            # Self-damage penalty: NEVER rob our own tiles
            self_touch = tile_nodes & my_nodes
            if self_touch:
                score = -1000.0 - len(self_touch) * 500.0
            else:
                pips = PIPS.get(tile.number, 0)
                score = pips * 10.0
                # Opponent building bonus
                opp_touch = tile_nodes & set(opp_nodes.keys())
                for on in opp_touch:
                    if opp_nodes[on] == CITY:
                        score += 40.0  # City blocks 2 cards per roll!
                    else:
                        score += 15.0

                # Target color bonus (must steal from opponent)
                if target_color is not None and target_color != obs.me:
                    score += 20.0

            if score > best_score:
                best_score = score
                best_i = i

        return best_i

    def _win_now(self, world, me, playable_actions):
        t0 = time.process_time()

        def dfs(game, depth):
            if time.process_time() - t0 > 0.008:
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
