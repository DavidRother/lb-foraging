from lbforaging.foraging.pettingzoo_environment import parallel_env
from lbforaging.foraging.manual_policy import ManualPolicy


players = 2
max_player_level = 2
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False

env = parallel_env(players=players, max_player_level=max_player_level, field_size=field_size, max_food=max_food,
                   sight=sight, max_episode_steps=max_episode_steps, force_coop=force_coop)

env.reset()
env.render()
env.unwrapped.screenshot()

terminations = {"player_0": False}

manual_policy = ManualPolicy(env, agent_id="player_0")

while not all(terminations.values()):
    action = {"player_0": manual_policy("player_0")}
    observations, rewards, terminations, truncations, infos = env.step(action)
    env.render()



