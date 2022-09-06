from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import gym

from lbforaging.foraging.environment import ForagingEnv


def env(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop, normalize_reward=True,
        grid_observation=False, penalty=0.0):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = ZooForagingEnvironment(players, max_player_level, field_size, max_food, sight, max_episode_steps,
                                      force_coop, normalize_reward, grid_observation, penalty)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class ZooForagingEnvironment(AECEnv):

    metadata = {'render.modes': ['human'], 'name': "foraging_zoo"}

    def __init__(self, players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop,
                 normalize_reward=True, grid_observation=False, penalty=0.0):

        super().__init__()
        self.foraging_env = ForagingEnv(players, max_player_level, field_size, max_food, sight, max_episode_steps,
                                        force_coop, normalize_reward, grid_observation, penalty)
        self.possible_agents = ["player_" + str(r) for r in range(players)]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {agent: self.foraging_env.get_observation_space() for agent in self.possible_agents}
        self.action_spaces = {agent: gym.spaces.Discrete(6) for agent in self.possible_agents}
        self.has_reset = True

        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_observation = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        self.t = 0
        self.last_rewards = [0, 0]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self):
        obs = self.foraging_env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.current_observation = {agent: obs[idx] for idx, agent in enumerate(self.agents)}

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.t = 0

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        obs, rewards, done, info = self.foraging_env.step(actions)
        self.last_rewards = rewards

        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done[idx]
            self.current_observation[agent] = obs[idx]
            self.rewards[agent] = rewards[idx]
            self.infos[agent] = info

    def observe(self, agent):
        returned_observation = self.current_observation[agent]
        return returned_observation

    def render(self, mode='human'):
        self.foraging_env.render(mode)

    def state(self):
        pass

    def close(self):
        self.foraging_env.close()
