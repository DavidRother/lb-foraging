import argparse
import logging
import random
import time
import gym
import numpy as np
import lbforaging
from lbforaging.foraging import zoo_environment


logger = logging.getLogger(__name__)
render = True

players = 2
max_player_level = 3
field_size = (8, 8)
max_food = 2
sight = 8
max_episode_steps = 50
force_coop = False
normalize_reward = True
grid_observation = False
penalty = 0.0

# env = gym.make("Foraging-8x8-2p-2f-v2")
env = zoo_environment.parallel_env(players=players, max_player_level=max_player_level, field_size=field_size,
                                   max_food=max_food, sight=sight, max_episode_steps=max_episode_steps,
                                   force_coop=force_coop, normalize_reward=normalize_reward,
                                   grid_observation=grid_observation, penalty=penalty)
obs = env.reset()

done = False

if render:
    env.render()
    time.sleep(0.5)

while not done:

    actions = {f"player_{idx}": env.action_spaces[f"player_{idx}"].sample() for idx in range(players)}

    nobs, nreward, ndone, _ = env.step(actions)
    if sum(nreward) > 0:
        print(nreward)

    if render:
        env.render()
        time.sleep(0.5)

    done = np.all(ndone)
# print(env.players[0].score, env.players[1].score)
