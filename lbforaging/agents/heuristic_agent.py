import random
import numpy as np
from lbforaging.foraging.agent import Agent
from lbforaging.foraging.environment import Action


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def _step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
	H1 agent always goes to the closest food
	"""

    name = "H1"

    def _step(self, obs):
        try:
            r, c = self._closest_food(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD
        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
	H2 Agent goes to the one visible food which is closest to the centre of visible players
	"""

    name = "H2"

    def _step(self, obs):

        players_center = self._center_of_players(obs.players)

        try:
            r, c = self._closest_food(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
	H3 Agent always goes to the closest food with compatible level
	"""

    name = "H3"

    def _step(self, obs):

        try:
            r, c = self._closest_food(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    """
	H4 Agent goes to the one visible food which is closest to all visible players
	 such that the sum of their and H4's level is sufficient to load the food
	"""

    name = "H4"

    def _step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        try:
            r, c = self._closest_food(obs, players_sum_level, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H5(HeuristicAgent):
    """
	H5 Agent always goes to the food in memory location 1 or 2 depending on the setting
	"""

    name = "H5"

    def _step(self, obs):

        # food_1 = obs.
        foods = list(zip(*np.nonzero(obs.field)))
        if len(foods) > 1:
            food_1 = foods[0]
            food_2 = foods[1]
        else:
            food_1 = foods[0]
            food_2 = foods[0]
        # food_choice = random.choices([food_1, food_2], weights=[0.25, 0.75], k=1)[0]
        food_choice = food_2
        # try:
        #     r, c = self._closest_food(obs, players_sum_level, players_center)
        # except TypeError:
        #     return random.choice(obs.actions)
        r, c = self._closest_food(obs)
        y, x = self.observed_position
        # print(f"Food position: {food_choice}")
        # print(f"Agent position: {y, x}")
        # print(f"Closest food: {r, c}")

        if (abs(food_choice[0] - y) + abs(food_choice[1] - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((food_choice[0], food_choice[1]), obs.actions)
        except ValueError:
            return random.choice(obs.actions)

