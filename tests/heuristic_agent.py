from lbforaging.foraging.pettingzoo_environment import parallel_env, ObservationSpace
from lbforaging.foraging.manual_policy import ManualPolicy
from lbforaging.agents.heuristic_agent import H1


players = 2
max_player_level = 2
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False
obs_spaces = [ObservationSpace.VECTOR_OBSERVATION, ObservationSpace.SYMBOLIC_OBSERVATION]

env = parallel_env(players=players, max_player_level=max_player_level, field_size=field_size, max_food=max_food,
                   sight=sight, max_episode_steps=max_episode_steps, force_coop=force_coop, obs_spaces=obs_spaces)

observations, infos = env.reset()
env.render()

terminations = {"player_0": False}

manual_policy = ManualPolicy(env, agent_id="player_0")
agent = H1("player_1")

while not all(terminations.values()):
    obs = observations["player_1"]
    act = agent.step(obs)
    action = {"player_0": manual_policy("player_0"), "player_1": act}
    observations, rewards, terminations, truncations, infos = env.step(action)
    env.render()



