import gym
import time
import random
import numpy as np


# set up and run the environment
def run():
    env = gym.make('Pong-v0')

    for i in range(1000):
        state = env.reset()
        epoch, penalty, reward = 0,0,0
        done = False

        # take a random action at each step
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            time.sleep(.01)
            if done:
                print('Finished episode {}'.format(i))
    env.close()

if __name__ == '__main__':
    run()
    print('Done Training')