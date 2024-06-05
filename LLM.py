import os
import sys
import yaml
import openai
import llmengine
from typing import Callable

llmengine.api_engine.api_key = os.environ["LLM_ENGINE_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

HUGGING_FACE_LLMs = []
LLM_ENGINE_LLMs = ['llama-2-70b-chat']
OPENAI_LLMs = ["davinci-002"]
OPENAI_CHAT_LLMs = ['gpt-3.5-turbo-0125']
all_llms = HUGGING_FACE_LLMs + LLM_ENGINE_LLMs + OPENAI_LLMs + OPENAI_CHAT_LLMs

class LLM:
    def __init__(self, model: str, interface: Callable):
        self.model = model
        self.interface = interface

    def __call__(self, prompt: str, stop: list):
        return self.interface(prompt, self.model, stop)

def get_llm(model: str) -> Callable:
    "gets the correct llm and api-interface for the model"
    if model in LLM_ENGINE_LLMs:
        return LLM(model, open_source_llm)
    if model in OPENAI_LLMs:
        return LLM(model, open_ai_llm)
    if model in OPENAI_CHAT_LLMs:
        return LLM(model, chat_llm)
    else: 
        raise NotImplementedError

# --------------------
# API Interfaces
# --------------------
def open_ai_llm(prompt, model = "davinci-002", stop=["\n"]):
    response = openai.Completion.create(
      model=model,
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]

def open_source_llm(message, model="llama-2-70b-chat", stop=["\n"]):
  response = llmengine.Completion.create(
    model=model,
    prompt=message,
    max_new_tokens=100,
    temperature=0,
    stop_sequences = stop
  )
  return response.output.text

def chat_llm(message, model = 'gpt-3.5-turbo-0125', role='assistant', stop=["\n"]):
    input = [{"role": role,"content": message}]
    response = openai.ChatCompletion.create(
      model = model,
      messages = input,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0].message['content']