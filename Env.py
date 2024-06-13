from typing import Iterable
import os
import yaml
import alfworld

def set_up_alfworld_env(config_file):
    os.environ["ALFWORLD_DATA"] = 'Alfworld'
    with open(f'Alfworld/{config_file}') as reader:
        config = yaml.safe_load(reader)
    split = "train"
    alfred_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = alfred_env.init_env(batch_size=1)
    return env

class EnvWrapper():
    "Wraps gym-like envs"
    def __init__(self, env):
        self.env = env

    def step(self, action: str):
        obs, reward, done, info = self.env.step([action]) # note there is support to play multiple games at once.
        obs, reward, done = obs[0], info['won'][0], done[0]
        return obs, reward, done, info
    
    def reset(self):
        obs, info = self.env.reset()
        return obs[0], info
    
class ReActEnv(EnvWrapper):
    "Wraps the Alfworld environment for ReAct"
    def __init__(self, env):
        super(ReActEnv, self).__init__(env)

    def step(self, action: str):
        obs, reward, done, info = self.env.step([action]) # note there is support to play multiple games at once.
        obs, reward, done = obs[0], info['won'][0], done[0]
        if action.startswith('think:'):
            obs = 'OK.'
        return obs, reward, done, info

class QuitEnv(EnvWrapper):
    "Ends the episode if the agent quits."
    def __init__(self, env):
        super(QuitEnv, self).__init__(env)
    def step(self, action: str):
        obs, reward, done, info = self.env.step([action]) # note there is support to play multiple games at once.
        obs, reward, done = obs[0], info['won'][0], done[0]
        if action.startswith('think:'):
            obs = 'OK.'
        if action.__contains__('quit'): #ADDED: quitting ends the game
            done = True
            reward = 0
            obs = 'Quit action recieved, ending task...'
        return obs, reward, done, info