from lbforaging.foraging.pettingzoo_environment import parallel_env, ObservationSpace
from lbforaging.foraging.manual_policy import ManualPolicy
from lbforaging.agents.heuristic_agent import H1, H2, H3
import time


players = 4
max_player_level = 2
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False
collisions = True
food_types = ["apple", "banana"]
tasks = ["collect_apples", "collect_apples", "collect_apples", "collect_apples"]
obs_spaces = [ObservationSpace.SYMBOLIC_OBSERVATION, ObservationSpace.SYMBOLIC_OBSERVATION,
              ObservationSpace.SYMBOLIC_OBSERVATION, ObservationSpace.SYMBOLIC_OBSERVATION]

env = parallel_env(players=players, max_player_level=max_player_level, field_size=field_size, max_food=max_food,
                   sight=sight, max_episode_steps=max_episode_steps, force_coop=force_coop, tasks=tasks,
                   obs_spaces=obs_spaces, collision=collisions, food_types=food_types, agent_respawn_rate=0.1,
                   grace_period=0, agent_despawn_rate=0.1)

observations, infos = env.reset()
env.render()

terminations = {"player_0": False}

manual_policy = ManualPolicy(env, agent_id="player_0")
agent0 = H1("player_0")
agent1 = H1("player_1")
agent2 = H2("player_2")
agent3 = H3("player_3")

accumulated_rewards = {"player_0": 0, "player_1": 0, "player_2": 0, "player_3": 0}
terminations = {"player_0": False, "player_1": False, "player_2": False, "player_3": False}
truncations = {"player_0": False, "player_1": False, "player_2": False, "player_3": False}
steps = 0
while True:
    # env.render()
    steps += 1
    actions = {}
    if "player_0" in observations and not truncations["player_0"] and not terminations["player_0"]:
        act = agent1.step(observations["player_0"])
        actions["player_0"] = act
    if "player_1" in observations and not truncations["player_1"] and not terminations["player_1"]:
        predator1_action = env.action_space("player_1").sample()
        actions["player_1"] = predator1_action
    if "player_2" in observations and not truncations["player_2"] and not terminations["player_2"]:
        predator2_action = env.action_space("player_2").sample()
        actions["player_2"] = predator2_action
    if "player_3" in observations and not truncations["player_3"] and not terminations["player_3"]:
        predator3_action = env.action_space("player_3").sample()
        actions["player_3"] = predator3_action
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    time.sleep(0.3)
    print(truncations)
    for agent_id in accumulated_rewards:
        accumulated_rewards[agent_id] += rewards[agent_id]

    if all(terminations.values()):
        break

    if all(truncations.values()):
        break



