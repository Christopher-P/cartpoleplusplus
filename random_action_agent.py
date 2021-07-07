#!/usr/bin/env python
import argparse
import random
import time

from cartpoleplusplus import CartPoleBulletEnv

# Params
use_gui = True
env = CartPoleBulletEnv(renders=use_gui)

# Number of episodes
nb_episodes = 200

for _ in range(nb_episodes):
    env.reset()
    done = False
    total_reward = 0
    steps = 0
    while True:
        action = 0#env.action_space.sample()
        _state, reward, done, info = env.step(action)
        print(_state)
        steps += 1
        total_reward += reward
        time.sleep(1/30)

print(total_reward)

env.reset()  # hack to flush last event log if required
