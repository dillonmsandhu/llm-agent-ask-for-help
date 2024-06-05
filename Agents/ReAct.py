from Agents.BaseAgent import Agent
from typing import Callable

class ReAct(Agent):
    "ReAct Agent designed to work in any AlfWorld task defined in the task-specific prompts"
    def __init__(self, llm: Callable, task_types: dict, prompt_file: dict):
        self.llm = llm
        self.task_types = task_types
        self.d = self.get_task_specific_prompts(prompt_file)
    
    def get_task_specific_prompts(self, prompt_file : str) -> dict:
        "Each kind of task in task_types has 2 example tasks provided."
        import json
        with open(prompt_file, 'r') as f:
            d = json.load(f)
        return d

    def get_task_name(self, info: dict):
        "Gets the task name to load"
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        return name

    def generate_context(self, task_name: str, quit_hint = '', to_print = False):
        "Generates the context for the agent based on the task-specific prompts and the task name."
        
        for (k, v) in self.task_types.items():
            if task_name.startswith(k):
                self.task_type = k
                self.context = 'Interact with a household to solve a task. ' + quit_hint + ' Here are two examples.\n' + self.d[f'react_{v}_1'] + self.d[f'react_{v}_0'] + '\nHere is the task.\n' 
                if to_print: print(self.context)
                return
        raise ValueError(f'Task {task_name} not found in task_types')
    
    def clean_obs(self, obs: str) -> str:
        return obs[obs.find('. ')+2:] if obs.startswith('You arrive at loc ') else obs

    def append_history(self, obs: str) -> str:
        "Concatentate context and observation to feed to the LLM."
        return self.context + self.clean_obs(obs)  + '\n>'
    
    def joint_policy(self, obs: str) -> str:
        """Joint policy for quitting and executing actions"""
        action = self.llm(self.append_history(obs), stop=['\n']).strip()
        if action == 'Nothing happens.':
            print("Action taken is nothing happens")
            print("observation: ", obs)
            with open('nothing happens action.txt', 'w') as f:
                print("Full observation" , self.append_history(obs), file = f)
            
        return action
    
    def update_context(self, obs: str, action = None) -> None:
        "Append the last transition to the context."
        if action:
            self.context += f' {action}\n{self.clean_obs(obs)}\n>'
        else:
            self.context += obs + '\n>'