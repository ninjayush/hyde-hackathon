"""Proxy for O: catanatron's SameTurnAlphaBeta depth 2, the strongest library bot
that fits the clock. O is stronger than this, so this is a floor."""
from catanatron.players.minimax import SameTurnAlphaBetaPlayer


class Agent:
    def __init__(self):
        self.p = None

    def decide(self, obs, playable_actions) -> int:
        if self.p is None:
            self.p = SameTurnAlphaBetaPlayer(obs.me, depth=2)
        import random
        world = obs.sample_world(random.Random(0))
        a = self.p.decide(world, playable_actions)
        for i, x in enumerate(playable_actions):
            if x.action_type == a.action_type and x.value == a.value:
                return i
        return 0
