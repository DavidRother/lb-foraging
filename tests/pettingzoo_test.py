from lbforaging.foraging.pettingzoo_environment import parallel_env, ObservationSpace
from lbforaging.foraging.manual_policy import ManualPolicy


players = 1
max_player_level = 2
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False
collisions = True
food_types = ["apple", "banana"]
tasks = ["collect_bananas"]
obs_spaces = [ObservationSpace.VECTOR_OBSERVATION]

env = parallel_env(players=players, max_player_level=max_player_level, field_size=field_size, max_food=max_food,
                   sight=sight, max_episode_steps=max_episode_steps, force_coop=force_coop, tasks=tasks,
                   obs_spaces=obs_spaces, collision=collisions, food_types=food_types, agent_respawn_rate=0.1,
                   grace_period=0, agent_despawn_rate=0.1)

env.reset()
env.render()

terminations = {"player_0": False}

manual_policy = ManualPolicy(env, agent_id="player_0")

while not all(terminations.values()):
    action = {"player_0": manual_policy("player_0")}
    observations, rewards, terminations, truncations, infos = env.step(action)
    print(observations)
    env.render()



