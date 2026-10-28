# Strategy Research & Architecture Blueprint: Achieving 90%+ Win Rate in 2-Player Catan

## Executive Summary

To achieve a 90%+ win rate against **Bot F** (Catanatron's baseline `ValueFunctionPlayer`) and outperform **Bot O** under the strict evaluation harness constraints (1 CPU core, ~1,000 ms per game, 2 GB RAM, imperfect information), we must systematically exploit F's structural weaknesses while staying strictly within the CPU budget.

The previous in-repo attempt reached only ~39.5% win rate because of critical bugs in opening settlement evaluation (scale mismatch and evaluating moves from the opponent's perspective). Fixing these and implementing a structured, multi-tier strategy will catapult performance.

---

## 1. Deconstructing the Opponent: Why Bot F Fails

Bot F is a 1-ply greedy searcher evaluating positions after a single legal action using a linear combination of 13 features:

```python
# F's 13 Features and Weights
W_F = {
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
}
```

### Critical Flaws in F:
1. **Single-World Blind Spot**: F calls `obs.sample_world()` exactly once per decision. It is completely blind to opponent development card distribution and hidden Victory Points (VPs).
2. **Intermediate Action Horizon Trap (1-Ply Myopia)**:
   - A Catan turn is a *sequence* of actions: e.g., `MARITIME_TRADE` -> `BUILD_SETTLEMENT` -> `END_TURN`.
   - At 1-ply, F evaluates the state *immediately after* `MARITIME_TRADE`. Giving away 4 cards for 1 drops `hand_resources` from 4 to 1 and drops `hand_synergy` before the settlement is built! F often avoids beneficial bank trades because the intermediate step looks worse.
3. **Zero Port Awareness**: F's evaluation contains no term for port ownership. In 2-player Catan with **no player-to-player trading**, 2:1 and 3:1 ports are decisive because trading 4:1 with the bank is extremely inefficient.
4. **Suboptimal Robber Placement**: F places the robber using raw production drop without considering the opponent's current bottleneck resource or targeting high-frequency hexes where the opponent has cities.
5. **No 10-VP Sprint / Endgame Awareness**: When 1-2 moves away from victory, F still optimizes intermediate features (like hand synergy or buildable nodes) rather than aggressively converting all resources/devs to reach 10 VPs.
6. **Ineffective Discard Management**: F's penalty for holding >7 cards is -5, whereas having 8 cards and losing 4 to a 7 roll destroys massive value ($4 \times 1 = 4$ resources, but losing half your hand when rolling 7 is devastating).

---

## 2. The Core Architecture for a 90%+ Win Rate Agent

To beat F reliably (and defeat O), our agent must combine:
1. **Ultra-Fast Heuristic Evaluation (`FastEval`)**
2. **Targeted Same-Turn Sequence Search (Macro Actions / Turn Flattening)**
3. **Smart Opening Settlement Engine (Initial Placement is 40% of the game)**
4. **Adaptive Time Budgeting & Multi-World Sampling**
5. **Heuristic Overrides for Robber, Discard, and Dev Cards**

---

## 3. Detailed Algorithmic Pillars

### Pillar 1: High-Win-Rate Opening Placement Engine
In 2-player Catan, initial settlements determine ~40-50% of the game outcome:
- **Total Production Dots**: High frequency tiles (6, 8 = 5 pips; 5, 9 = 4 pips; 4, 10 = 3 pips).
- **Resource Variety**: Having access to all 5 resources is worth significantly more than having 5 tiles of sheep. A balanced 4-5 resource spread prevents stalling.
- **Ore-Wheat-Sheep (OWS) vs Wood-Clay Engine**:
  - In 2-player Catan, Cities + Dev Cards (OWS) scales exponentially faster than pure road expansion. Cities double tile production without taking up new board space.
- **Port Synergy**: An 8-ore tile adjacent to an Ore 2:1 port is an instant win condition (convert 2 Ore into any resource).
- **Settlement 2 Strategy**: 
  - Settlement 1: Maximize raw production pips + OWS / Wood-Brick pairs.
  - Settlement 2: Fill missing resources from Settlement 1, secure a matching 2:1 port or 3:1 port, and point the road toward a high-value expansion node.

### Pillar 2: Addressing the 1-Ply Horizon Trap (Macro-Actions & Turn Pruning)
Why pure Alpha-Beta or MCTS fails in Catan:
- Branching factor is enormous (maritime trades generate dozens of actions per turn).
- As shown in `PROBLEM_STATEMENT.md`, `SameTurnAlphaBetaPlayer` timed out in 415/1000 games and won only 8%!
- **The Solution**: **Macro-Action Flattening / Selective Lookahead**:
  - Don't search full game trees.
  - For `MARITIME_TRADE`, evaluate the trade *bundled* with what it immediately enables (e.g., Trade 4 Sheep for 1 Wheat -> Build Settlement).
  - If a sequence leads to a settlement, city, or development card within the current turn, evaluate the state *at the end of that sequence*.

### Pillar 3: Specialized Action Heuristics

#### 1. Robber Placement (`MOVE_ROBBER`)
Instead of generic lookahead:
- Target the hex that minimizes opponent production while maximizing our relative advantage.
- Specifically:
  - Never block a hex where we also have a settlement/city.
  - Prioritize opponent's **Cities** (2 production points blocked per roll instead of 1).
  - Prioritize the opponent's **scarce resource** (e.g., if opponent has no other Wheat tile, blocking their 8 Wheat completely starves their city/settlement builds).
  - Steal from the opponent (always choose `victim_color = opp`).

#### 2. Discarding (`DISCARD_RESOURCE`)
When a 7 is rolled and we hold >7 cards:
- Identify our immediate build goal (City = 3 Ore + 2 Wheat; Settlement = 1 Wood + 1 Brick + 1 Sheep + 1 Wheat).
- Discard surplus resources that don't contribute to the immediate goal or where we have high production.
- Retain bottleneck cards that are hard to produce.

#### 3. Development Card Purchase & Play
- Dev cards are high-leverage:
  - **Knights**: Protect our best hexes, steal cards, and secure the 2 VP Largest Army.
  - **Road Building**: Instant 2 roads = saves 2 Wood + 2 Brick (4 resources!) and secures Longest Road.
  - **Year of Plenty**: Instant 2 targeted resources to complete a City or Settlement.
  - **Monopoly**: Devastating when timed after opponent accumulates a specific resource.
  - **Victory Points**: Hidden until game over; 5 of 25 cards are VPs!

### Pillar 4: Feature Engineering for `FastEval`

Instead of relying solely on Catanatron's heavy `base_fn` (which spends time on graph reachability), we augment it with targeted features:

| Feature | Weight Scale | Purpose |
|---|---|---|
| `public_vps` | 3e14 | Instant winning priority |
| `actual_vps` (our hidden devs) | 3e14 | Recognize when we can claim victory |
| `production` | 1.2e8 | Total expected pips/turn |
| `enemy_production` | -1.0e8 | Deny opponent production |
| `production_variety` | 2.5e7 | Bonus for holding access to 3, 4, 5 distinct resources |
| `port_synergy` | 3.0e7 | Production of resource R * having 2:1 port of R |
| `general_port` | 8.0e6 | 3:1 port value |
| `city_readiness` | 5.0e4 | Proximity to City in hand (Ore + Wheat count) |
| `settlement_readiness` | 3.0e4 | Proximity to Settlement in hand |
| `army_lead` | 1.5e3 | Differential in played knights toward Largest Army |
| `road_lead` | 1.0e3 | Differential in road length toward Longest Road |
| `discard_danger` | -50.0 | Heavy penalty per card above 7 before ending turn |

---

## 4. Benchmark Performance Target: Path to 90%+

| Version | Win Rate vs F | Mean CPU/game | Status |
|---|---|---|---|
| **v1 (Baseline in repo)** | **39.5%** | 375 ms | Broken opening & scale bugs |
| **v2 (Bugfix + Heuristic Opening + Port Synergy)** | **65% - 75%** | 320 ms | Fixes opening scale, adds ports |
| **v3 (Multi-World 3x + Macro Trade Bundling + Robber Logic)** | **82% - 88%** | 450 ms | Eliminates intermediate trade myopia |
| **v4 (Optimized Weights via Tuner + Endgame Sprint)** | **90%+** | 500 ms | Dominant win rate, well inside 1.5s cap |
