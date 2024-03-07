import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gymnasium import Env
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class ObservationSpace(Enum):
    GRID_OBSERVATION = 0
    VECTOR_OBSERVATION = 1
    SYMBOLIC_OBSERVATION = 2
    GLOBAL_GRID_OBSERVATION = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.task = None

    def setup(self, position, level, field_size, task):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0
        self.task = task

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        tasks=None,
        normalize_reward=True,
        penalty=0.0,
        obs_spaces=None,
        reward_scheme="new",
        collision=False,
        food_types=None,
        agent_respawn_rate=0.0,
        grace_period=20,
        agent_despawn_rate=0.0
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]
        self.obs_spaces = obs_spaces or [ObservationSpace.VECTOR_OBSERVATION] * players
        self.field_size = field_size

        apple_field = np.zeros(field_size, np.int32)
        banana_field = np.zeros(field_size, np.int32)
        self.food_type_mapping = {"apple": apple_field, "banana": banana_field}
        self.task_mapping = defaultdict(str, {"collect_apples": "apple", "collect_bananas": "banana"})

        self.penalty = penalty
        self.food_types = food_types or ["apple"]
        
        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None

        self.n_agents = len(self.players)
        self.last_actions = [Action.NONE] * self.n_agents
        self.current_step = 0
        self.tasks = tasks or (["collect_apples"] * self.n_agents)
        self.np_random = None
        self.reward_scheme = reward_scheme
        self.collision = collision

        self.agent_respawn_rate = agent_respawn_rate
        self.grace_period = grace_period
        self.agent_despawn_rate = agent_despawn_rate
        self.agent_grace_period = [self.grace_period] * self.n_agents
        self.active_agents = [True] * self.n_agents
        self.status_changed = [False] * self.n_agents
        self.relevant_agents = self.players

    def reset(self, **kwargs):
        apple_field = np.zeros(self.field_size, np.int32)
        banana_field = np.zeros(self.field_size, np.int32)
        self.food_type_mapping = {"apple": apple_field, "banana": banana_field}
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])
        self.last_actions = [Action.NONE for _ in self.players]

        self.spawn_food(self.max_food, max_level=sum(player_levels[:3]))
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        self.agent_grace_period = [self.grace_period] * self.n_agents
        self.active_agents = [True] * self.n_agents
        self.status_changed = [False] * self.n_agents
        self.relevant_agents = self.players

        nobs, nreward, ndone, ntruncated, ninfo = self._make_gym_obs(self.relevant_agents)
        return nobs, ninfo

    def step(self, actions):
        active_agents_start = [agent for idx, agent in enumerate(self.players) if self.active_agents[idx]]
        self.last_actions = actions
        self.clear_rewards()
        assert len(actions) == len(active_agents_start)
        self.status_changed = [False] * self.n_agents
        self.current_step += 1
        self.do_world_step(actions, active_agents_start)
        self.handle_agent_spawn()
        self.relevant_agents = self.compute_relevant_agents()
        relevant_fields = [self.food_type_mapping[self.task_mapping[task]] for task in self.tasks
                           if self.task_mapping[task] in self.food_type_mapping]
        self._game_over = sum([field.sum() for field in relevant_fields]) == 0
        self._gen_valid_moves()
        self.compute_rewards()
        nobs, nreward, ndone, ntruncated, ninfo = self._make_gym_obs(active_agents_start)
        return nobs, nreward, ndone, ntruncated, ninfo

    def clear_rewards(self):
        for p in self.players:
            p.reward = 0

    def compute_rewards(self):
        if self.reward_scheme == "new":
            for p in self.players:
                p.reward *= 10
                p.reward -= 0.3
                p.score += p.reward
        elif self.reward_scheme == "old":
            for p in self.players:
                p.score += p.reward
        else:
            raise Exception("No valid reward scheme selected!")

    def do_world_step(self, actions, active_agents_start):
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(active_agents_start, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(active_agents_start, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info("{}{} attempted invalid action {}.".format(player.name, player.position, action))
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(active_agents_start, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if self.collision:
                if len(v) > 1:  # make sure no more than one player will arrive at location
                    continue
                else:
                    v[0].position = k
            else:
                for p in v:
                    p.position = k
        self.handle_loading(loading_players, active_agents_start)

    def handle_loading(self, loading_players, active_agents_start):
        while loading_players:
            player = loading_players.pop()
            for food_type in self.food_types:
                field = self.food_type_mapping[food_type]
                food_location = self.adjacent_food_location(*player.position, field)
                if food_location is None:
                    continue
                frow, fcol = food_location
                food = field[frow, fcol]

                adj_players = self.adjacent_players(frow, fcol, active_agents_start)
                adj_players = [p for p in adj_players if p in loading_players or p is player]

                adj_player_level = sum([a.level for a in adj_players])

                loading_players = loading_players - set(adj_players)

                if adj_player_level < food:
                    # failed to load
                    for a in adj_players:
                        a.reward -= self.penalty
                else:
                    for a in adj_players:
                        field[frow, fcol] = 0
                        if food_type != self.task_mapping[a.task]:
                            continue
                        a.reward = float(a.level * food)
                        if self._normalize_reward:
                            a.reward = a.reward / float(adj_player_level * self._food_spawned)  # normalize reward

    def handle_agent_spawn(self):
        for i in range(self.n_agents):
            if self.agent_grace_period[i] > 0:
                self.agent_grace_period[i] -= 1
            else:
                # active = self.active_agents[i]
                if self.active_agents.count(True) > 1 and self.active_agents[i] \
                   and np.random.random() < self.agent_despawn_rate:
                    self.despawn_agent(i)
                elif not self.active_agents[i] and np.random.random() < self.agent_respawn_rate:
                    self.respawn_agent(i)

    def compute_relevant_agents(self):
        return [agent for idx, agent in enumerate(self.players) if self.active_agents[idx] or self.status_changed[idx]]

    def despawn_agent(self, index):
        self.active_agents[index] = False
        self.status_changed[index] = True

    def respawn_agent(self, index):
        self.active_agents[index] = True
        self.status_changed[index] = True
        self.agent_grace_period[index] = self.grace_period
        self.reset_player_location(self.players[index])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_player_location(self, player):
        attempts = 0
        while attempts < 1000:
            row = self.np_random.integers(0, self.rows)
            col = self.np_random.integers(0, self.cols)
            if all([self._is_empty_location(row, col, field) for field in self.food_type_mapping.values()]):
                player.position = (row, col)
                break
            attempts += 1

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if self.obs_spaces[0] == ObservationSpace.VECTOR_OBSERVATION:
            field_x = self.field_size[1]
            field_y = self.field_size[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)

            min_obs = [0, 0, 0, 0] * max_food * len(self.food_types) + [0, 0, 0] * len(self.players)
            max_obs = [field_x-1, field_y-1, max_food_level, 1] * max_food * len(self.food_types) + \
                      [field_x-1, field_y-1, self.max_player_level] * len(self.players)

        elif self.obs_spaces[0] == ObservationSpace.GLOBAL_GRID_OBSERVATION:
            # grid observation space
            grid_shape = (1 + 2 * self.field_size[0], 1 + 2 * self.field_size[1])

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min]).transpose((1, 2, 0))
            max_obs = np.stack([agents_max, foods_max, access_max]).transpose((1, 2, 0))
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min]).transpose((1, 2, 0))
            max_obs = np.stack([agents_max, foods_max, access_max]).transpose((1, 2, 0))

        shape = np.array(min_obs).shape
        return gym.spaces.Box(np.array(min_obs, dtype=np.float32), np.array(max_obs, dtype=np.float32),
                              shape=shape, dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape, "collect_apples")
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()
        return env

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, field, distance=1, ignore_diag=False):
        if not ignore_diag:
            return field[
                max(row - distance, 0): min(row + distance + 1, self.rows),
                max(col - distance, 0): min(col + distance + 1, self.cols),
            ]

        return (
            field[max(row - distance, 0): min(row + distance + 1, self.rows), col].sum()
            + field[row, max(col - distance, 0): min(col + distance + 1, self.cols)].sum()
        )

    def adjacent_food(self, row, col, field):
        return (
            field[max(row - 1, 0), col]
            + field[min(row + 1, self.rows - 1), col]
            + field[row, max(col - 1, 0)]
            + field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col, field):
        if row > 1 and field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col, players):
        return [player for player in players if abs(player.position[0] - row) == 1 and player.position[1] == col
                or abs(player.position[1] - col) == 1 and player.position[0] == row]

    def spawn_food(self, max_food, max_level):
        min_level = max_level if self.force_coop else 1

        for food_type in self.food_types:
            field = self.food_type_mapping[food_type]
            food_count = 0
            attempts = 0
            while food_count < max_food and attempts < 1000:
                attempts += 1
                row = self.np_random.integers(1, self.rows - 1)
                col = self.np_random.integers(1, self.cols - 1)

                # check if it has neighbors:
                if not self.check_neighborhood(row, col):
                    continue

                field[row, col] = (min_level if min_level == max_level
                                   else self.np_random.integers(min_level, max_level))
                food_count += 1
            self._food_spawned += field.sum()

    def check_neighborhood(self, row, col):
        free = True
        for field in self.food_type_mapping.values():
            free = free and self.neighborhood(row, col, field).sum() == 0
            free = free and self.neighborhood(row, col, field, distance=2, ignore_diag=True) == 0
            free = free and self._is_empty_location(row, col, field)
            if not free:
                return free
        return free


    def _is_empty_location(self, row, col, field):
        if field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
        return True

    def spawn_players(self, max_player_level):
        for player, task in zip(self.players, self.tasks):

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                is_empty = [self._is_empty_location(row, col, self.food_type_mapping[field_type])
                            for field_type in self.food_types]
                if is_empty:
                    player.setup((row, col), self.np_random.integers(1, max_player_level + 1), self.field_size, task)
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return player.position[0] > 0 and self.check_empty(player.position[0] - 1, player.position[1])
        elif action == Action.SOUTH:
            return player.position[0] < self.rows - 1 and self.check_empty(player.position[0] + 1, player.position[1])
        elif action == Action.WEST:
            return player.position[1] > 0 and self.check_empty(player.position[0], player.position[1] - 1)
        elif action == Action.EAST:
            return player.position[1] < self.cols - 1 and self.check_empty(player.position[0], player.position[1] + 1)
        elif action == Action.LOAD:
            return sum([self.adjacent_food(*player.position, self.food_type_mapping[field_type])
                        for field_type in self.food_types]) > 0
        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def check_empty(self, row, col):
        return all([self._is_empty_location(row, col, self.food_type_mapping[field_type])
                    for field_type in self.food_types])

    def _transform_to_neighborhood(self, center, sight, position):
        return position[0] - center[0] + min(sight, center[0]), position[1] - center[1] + min(sight, center[1])

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player, task=""):
        players = []
        for a in self.relevant_agents:
            if (min(self._transform_to_neighborhood(player.position, self.sight, a.position)) >= 0) and \
               max(self._transform_to_neighborhood(player.position, self.sight, a.position)) <= 2 * self.sight:

                player_position = self._transform_to_neighborhood(player.position, self.sight, a.position)
                player_obs = self.PlayerObservation(position=player_position, level=a.level, is_self=a == player,
                                                    history=a.history, reward=a.reward if a == player else None)
                players.append(player_obs)
        fields = [np.copy(self.neighborhood(*player.position, self.food_type_mapping[field_type], self.sight))
                  for field_type in self.food_types]

        return self.Observation(
            actions=self._valid_actions[player],
            players=players,
            field=fields,
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def make_obs_array(self, observation):
        obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
        # obs[: observation.field.size] = observation.field.flatten()
        # self player is always first
        seen_players = [p for p in observation.players if p.is_self] + \
                       [p for p in observation.players if not p.is_self]

        for i in range(self.max_food * len(self.food_types)):
            obs[4 * i] = 0
            obs[4 * i + 1] = 0
            obs[4 * i + 2] = 0
            obs[4 * i + 3] = 0

        for idx, field in enumerate(observation.field):
            counter = 0
            for y, x in zip(*np.nonzero(field)):
                obs[4 * counter + (idx * self.max_food * 4)] = y
                obs[4 * counter + (idx * self.max_food * 4) + 1] = x
                obs[4 * counter + (idx * self.max_food * 4) + 2] = field[y, x]
                obs[4 * counter + (idx * self.max_food * 4) + 3] = 1
                counter += 1

        for i in range(len(self.players)):
            obs[self.max_food * len(self.food_types) * 4 + 3 * i] = -1
            obs[self.max_food * len(self.food_types) * 4 + 3 * i + 1] = -1
            obs[self.max_food * len(self.food_types) * 4 + 3 * i + 2] = 0

        for i, p in enumerate(self.players):
            if not self.active_agents[i]:
                continue
            obs[self.max_food * len(self.food_types) * 4 + 3 * i] = p.position[0]
            obs[self.max_food * len(self.food_types) * 4 + 3 * i + 1] = p.position[1]
            obs[self.max_food * len(self.food_types) * 4 + 3 * i + 2] = p.level

        return obs

    def make_global_grid_arrays(self):
        """
        Create global arrays for grid observation space
        """
        grid_shape_x, grid_shape_y = self.field_size
        grid_shape_x += 2 * self.sight
        grid_shape_y += 2 * self.sight
        grid_shape = (grid_shape_x, grid_shape_y)

        agents_layer = np.zeros(grid_shape, dtype=np.float32)
        for player in self.players:
            player_x, player_y = player.position
            agents_layer[player_x + self.sight, player_y + self.sight] = player.level
        food_layer = []
        for field in self.food_type_mapping.values():
            food_layer.append(np.zeros(grid_shape, dtype=np.float32))
            food_layer[-1][self.sight:-self.sight, self.sight:-self.sight] = field.copy()

        access_layer = np.ones(grid_shape, dtype=np.float32)
        # out of bounds not accessible
        access_layer[:self.sight, :] = 0.0
        access_layer[-self.sight:, :] = 0.0
        access_layer[:, :self.sight] = 0.0
        access_layer[:, -self.sight:] = 0.0
        # agent locations are not accessible
        for player in self.players:
            player_x, player_y = player.position
            access_layer[player_x + self.sight, player_y + self.sight] = 0.0
        # food locations are not accessible
        for field in self.food_type_mapping.values():
            food_x, food_y = field.nonzero()
            for x, y in zip(food_x, food_y):
                access_layer[x + self.sight, y + self.sight] = 0.0

        return np.stack([agents_layer, *food_layer, access_layer]).transpose((1, 2, 0))

    def get_agent_grid_bounds(self, agent_x, agent_y):
        return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1

    def get_player_reward(self, observation):
        for p in observation.players:
            if p.is_self:
                return p.reward

    def _make_gym_obs(self, active_agents_start):
        observations = [self._make_obs(player) for player in self.relevant_agents]
        nobs = []
        filtered_obs_spaces = [obs_space for obs_space, player in zip(self.obs_spaces, self.players)
                               if player in self.relevant_agents]
        for obs_space, player, obs in zip(filtered_obs_spaces, self.relevant_agents, observations):
            layers = self.make_global_grid_arrays()
            if obs_space == ObservationSpace.GRID_OBSERVATION:
                start_x, end_x, start_y, end_y = self.get_agent_grid_bounds(*player.position)
                nobs.append(layers[:, start_x:end_x, start_y:end_y])
            elif obs_space == ObservationSpace.VECTOR_OBSERVATION:
                nobs.append(self.make_obs_array(obs))
            if obs_space == ObservationSpace.GLOBAL_GRID_OBSERVATION:
                nobs.append(layers)
            else:
                nobs.append(obs)
        nreward = [self.get_player_reward(obs) for obs in observations]
        ndone = self.compute_terminations(observations)
        # ninfo = [{'observation': obs} for obs in observations]
        curated_actions = self._compute_curated_actions(active_agents_start)
        ninfo = self.compute_info(curated_actions)

        ntruncated = self.compute_truncated()
        return nobs, nreward, ndone, ntruncated, ninfo

    def compute_truncated(self):
        if self.current_step >= self._max_episode_steps:
            truncated = [True] * len(self.relevant_agents)
            self.active_agents = [False] * self.n_agents
            self.status_changed = [True if agent in self.relevant_agents else False for agent in self.players]
        else:
            truncated = [False] * len(self.relevant_agents)

        offset_idx = 0
        for idx, agent in enumerate(self.players):
            if agent not in self.relevant_agents:
                offset_idx += 1
                continue
            if self.status_changed[idx] and not self.active_agents[idx]:
                truncated[idx - offset_idx] = True
        return truncated

    def compute_terminations(self, observations):
        ndone = [obs.game_over for obs in observations]
        done = []
        idx = 0
        for player in self.players:
            if player not in self.relevant_agents:
                idx += 1
                continue
            done.append(ndone.pop(0))
            self.status_changed[idx] = done[-1]
            self.active_agents[idx] = not done[-1]
            idx += 1
        return done

    def _compute_curated_actions(self, active_agents_start):
        curated_actions = []
        idx_offset = 0
        for idx, player in enumerate(self.players):
            if player not in self.relevant_agents:
                idx_offset += 1
                continue
            if player in active_agents_start:
                curated_actions.append(self.last_actions[idx - idx_offset])
            else:
                idx_offset += 1
                curated_actions.append(0)
        return curated_actions

    def compute_info(self, curated_actions):
        # ninfo = [{"action": curated_actions[idx], "task": self.tasks[idx]}
        #          for idx, player in enumerate(self.relevant_agents)]
        ninfo = []
        idx_offset = 0
        for idx, player in enumerate(self.players):
            if player not in self.relevant_agents:
                idx_offset += 1
            else:
                ninfo.append({"action": curated_actions[idx - idx_offset], "task": self.tasks[idx - idx_offset]})
        return ninfo

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def save_image(self, path="screenshot.png"):
        self.viewer.save_image(path)
