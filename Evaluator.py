import sys
from Agents.BaseAgent import Agent

class Evaluator():
    
    def __init__(self, agent: Agent, env, num_eps: int, stored_run: dict):
        self.agent = agent
        self.env = env
        self.num_eps = num_eps
        self.stored_run = stored_run
    
    def __call__(self, quit_hint = '', to_print=False):
        "Evaluates the agent in the environment."        
        for n in range(self.num_eps):
            if n == 1 or n % 100: print(f"Running epsiode {n}")
            try:        
                r, t = self.play_alfworld(quit_hint, to_print = to_print)
            except:
                assert self.play_alfworld(quit_hint, to_print = to_print) == 'error'
                print('error running episode ', n)
                break
        return n
            
    def get_task_name(self, info: dict):
        # TODO: Probably this belongs with env
        "Gets the task name to load"
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        return name

    def play_alfworld(self, quit_hint='', to_print=False, max_steps=25):
        # TODO: Probably this belongs with env
        "Plays a game of alfworld, storing as a row in the stored_run dict."
        
        history = ''
        # generate few-shot examples based on task name.
        obs, info = self.env.reset() # The first observation includes the task specification
        obs = '\n'.join(obs.split('\n\n')[1:])
        history += obs
        name = self.get_task_name(info)
        
        self.agent.generate_context(name, quit_hint) # generate few-shot examples based on task name.
        self.agent.update_context(obs) # add the initial observation     
        
        self.stored_run['task_name'].append(name)
        self.stored_run['task_type'].append(self.agent.task_type)

        if to_print:
            print(obs)
            sys.stdout.flush()

        for i in range(1, max_steps+1):
            try:
                action = self.agent.joint_policy(obs)
            except:
                print("Error hitting API")
                return 'error'
            
            if action == 'Nothing happens.':
                print("Action taken is nothing happens")
                exit()
            obs, reward, done, info = self.env.step(action) # note there is support to play multiple games at once.
            self.agent.update_context(obs, action) # record the observation and action in internal state.
            history += f'Act {i}: {action}\nObs {i}: {self.agent.clean_obs(obs)}\n'
            if done:
                self.stored_run['task_success'].append(info['won'][0])
                self.stored_run['trajectory'].append(history) 
                self.stored_run['task_length'].append(i)
                return (reward, i)
        
        self.stored_run['task_success'].append(info['won'][0])
        self.stored_run['trajectory'].append(history)
        self.stored_run['task_length'].append(i) 
        return (0, i)