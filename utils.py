import pickle
import os
import yaml
import alfworld
import LLM
from Agents.ReAct import ReAct
import alfworld.agents.environment

def save_run(write_folder, file_name, stored_run):
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    
    write_file = write_folder + file_name
    print("Writing to ", write_file)
    mode = 'ab' if os.path.exists(write_file) else 'wb'
    file = open(write_file, mode)
    pickle.dump(stored_run, file)
    file.close()