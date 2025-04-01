import gymnasium as gym
from time import sleep
import torch
from numpy import sin, cos
from nn0 import NN

def get_angular(env):
    return torch.tensor(env.unwrapped.state, dtype=torch.float)

def get_cartesian(env):
    s = env.unwrapped.state
    p1 = [
        -env.unwrapped.LINK_LENGTH_1 * cos(s[0]),
        env.unwrapped.LINK_LENGTH_1 * sin(s[0]),
        ]
    p2 = [
        p1[0] - env.unwrapped.LINK_LENGTH_2 * cos(s[0] + s[1]),
        p1[1] + env.unwrapped.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

    return torch.tensor(p1+p2, dtype=torch.float)
    

if __name__ == '__main__':
    # ---------------
    # Definitions
    # ---------------
    max_it = 10

    nn = NN()
    # ---------------
    # Main loop 
    # ---------------
    env = gym.make('Acrobot-v1', render_mode='human')

    observation, info = env.reset()

    episode_over = False
    it = 0
    while (not episode_over) and (it < max_it):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        goal = get_cartesian(env)
        angles = get_angular(env) 
        pos = nn(angles) 
        loss = torch.norm(pos-goal)
        print(loss)
        episode_over = terminated or truncated
        it += 1
        sleep(0.05)
    env.close()
