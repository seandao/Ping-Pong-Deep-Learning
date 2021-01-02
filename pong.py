import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40)
import tensorflow as tf
import time
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import InputLayer
from keras.optimizers import Adam
import math
import glob
import io
import base64
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

# set up and run the environment
def run():
    # set up display recording
    display = Display(visible=0, size=(1400, 900))
    display = Display().start()

    env = wrap_env(gym.make('Pong-v0'))
    model = make_model()

    history = []
    obs = env.reset()
    prev_state = None
    done = False

    # declaring possible actions
    UP = 2
    DOWN = 3
    
    # hyperparameters
    gamma = 0.99

    # variables
    reward_sum = 0
    episode = 0
    x_train, y_train, rewards = [],[],[]

    # train model
    while True:
        cur_state = preproc_image(obs)

        if prev_state is not None:
            x = cur_state-prev_state
        else:
            x = np.zeros(80*80)
        
        prev_state = cur_state

        # policy network based on probability distribution
        prob = model.predict(np.expand_dims(x,axis=1).T)

        # decide what action to take
        if np.random.uniform() < prob:
            action = UP
        else:
            action = DOWN

        # outputs
        if action == 2:
            y = 1
        else:
            y = 0

        env.render()

        # record the inputs and labels
        x_train.append(x)
        y_train.append(y)

        #print('Taking action {}'.format(y))
        # step through environment
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        reward_sum += reward

        # if an episode is done, fit the model
        if done:
            print('Episode {} total reward was {}'.format(episode,reward_sum))
            history.append(reward_sum)

            if episode >= 3000 and reward_sum >= -12:
                break
            else:
                episode += 1
                # train model
                model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards,gamma))

                # reset variables for next episode
                x_train, y_train, rewards = [],[],[]
                obs = env.reset()
                reward_sum = 0
                prev_input = None

    plt.plot(history)
    plt.show()

    env.close()
    show_video()

# compile machine learning model
def make_model():
    model = Sequential()

    # add hidden layer 200 units
    model.add(Dense(units=200, input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

    # output layer sigmoid, move up or down
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    # compile model with optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return model

# make frame smaller and remove color
def preproc_image(fr):
    fr = fr[35:195]
    fr = fr[::2,::2,0]
    # remove backgrounds
    fr[fr == 144] = 0
    fr[fr == 109] = 0
    fr[fr != 0] = 1
    return fr.astype(np.float).ravel()

# normalize rewards
def discount_rewards(rewards, gamma):
    rewards = np.array(rewards)
    discount_r = np.zeros_like(rewards)
    sum = 0

    for i in reversed(range(0, rewards.size)):
        # if game ends then reset
        if rewards[i] != 0:
            sum = 0
        sum = sum * gamma + rewards[i]
        discount_r[i] = sum
        discount_r -= np.mean(discount_r) # normalize
        discount_r /= np.std(discount_r)
    return discount_r

# video recording of an episode
def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
            loop controls style="height: 400px;">
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")

# wrap the environment to use video
def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

if __name__ == '__main__':
    run()
    print('Done Training')