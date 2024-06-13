from typing import Callable
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, llm: Callable[[str], str]):
        self.llm = llm
    
    @abstractmethod
    def joint_policy(self, obs = None) -> str:
        "Use the LLM to return either an action or quit given an observation"
        pass
