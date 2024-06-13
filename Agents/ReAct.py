from Agents.BaseAgent import Agent
from typing import Callable, Iterable

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

    def generate_context(self, task_name: str, initial_obs: str, quit_hint = '', to_print = False):
        "Generates the context for the agent based on the task-specific prompts and the task name."
        
        for (k, v) in self.task_types.items():
            if task_name.startswith(k):
                self.task_type = k
                self.context = 'Interact with a household to solve a task. ' + quit_hint \
                                + ' Here are two examples.\n' + self.d[f'react_{v}_1'] + self.d[f'react_{v}_0'] \
                                + '\nHere is the task.\n' \
                                + initial_obs + '\n>'
                if to_print: print(self.context)
                return
        raise ValueError(f'Task {task_name} not found in task_types')
    
    def clean_obs(self, obs: str) -> str:
        return obs[obs.find('. ')+2:] if obs.startswith('You arrive at loc ') else obs

    def append_history(self, obs: str) -> str:
        "Concatentate context and observation to feed to the LLM."
        return self.context + self.clean_obs(obs)  + '\n>'
    
    def joint_policy(self, obs = None) -> str:
        """Joint policy for quitting and executing actions"""
        action = self.llm(self.context, stop=['\n']).strip()
        if action == 'Nothing happens.':
            print("Action taken is nothing happens")
            print("observation: ", obs)
            with open('nothing happens action.txt', 'w') as f:
                print("Full observation" , self.append_history(obs), file = f)
            
        return action
        
    def update_context(self, obs = None, action = None) -> None:
        "Append the last transition to the context."
        if action:
            self.context += f'{action}\n'
        if obs:
            self.context += self.clean_obs(obs) + '\n>'

class ChatReAct(ReAct):
    def __init__(self, llm: Callable, task_types: dict, prompt_file: dict):
        super().__init__(llm, task_types, prompt_file)
    
    def generate_context(self, task_name: str, initial_obs: str, quit_hint = '', to_print = False):
        """
        Generates the context for the agent based on the task-specific prompts and the task name.
        self.context will be a list of past interactions, initialized as:
            # user: 'Interact with a household to solve a task. ' + quit_hint + ' Here are two examples.\n'
            # example 1: _, a, u, a, _
            # example 2: _, a, u, a, _
            # user: + '\nHere is the task.\n' 
        """    
        def contiguous_example_roles(example):
            "Examples start and end with user content, but text must switch roles between user and assistant."
            ex = example
            if example[0]['role'] == 'user':
                line1 = example[0]['content']
                ex = example[1:]
            if example[-1]['role'] == 'user':
                final_line = example[-1]['content']
                ex = ex[:-1]
            return line1, final_line, ex
        
        for (k, v) in self.task_types.items():
            if task_name.startswith(k):
                self.task_type = k
                example1, example2 = self.d[f'react_{v}_1'], self.d[f'react_{v}_0']
                line1, final_line1, example1 = contiguous_example_roles(example1)
                line2, final_line2, example2 = contiguous_example_roles(example2)
                self.context = \
                    [{'role': 'user', 'content': 'Interact with a household to solve a task. ' + quit_hint + ' Here is an example.\n' + line1}] + example1 \
                    + [{"role": "user", "content": final_line1 + '\n' + line2}] + example2 \
                    + [{"role": "user", "content": final_line2 + '\nHere is your task: \n' + initial_obs}]
                if to_print: 
                    print(self.context)
                return
        raise ValueError(f'Task {task_name} not found in task_types')
    
    def update_context(self, obs = None, action = None) ->  None:
        "Append the last transition to the context."
        if action:
            self.context += [{"role": "assistant", "content": action}]
        if obs:
            self.context += [{"role": "user", "content": self.clean_obs(obs)}]
            
    def joint_policy(self, obs = None) -> str:
        """Joint policy for quitting and executing actions"""
        return self.llm(self.context, stop=['\n']).strip() 


