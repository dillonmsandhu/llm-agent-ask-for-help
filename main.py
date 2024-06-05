import LLM
import argparse
from datetime import datetime
from Agents.ReAct import ReAct
import Env
from Evaluator import Evaluator
import utils

parser = argparse.ArgumentParser(description='ReAct Agent')
parser.add_argument("--llm", type=str, default="llama-2-70b-chat", choices = LLM.all_llms, help="LLM to use")
parser.add_argument("--agent", type=str, default="ReAct", choices = LLM.all_llms, help="LLM agent wrapper to use")
parser.add_argument("--env", type=str, default="AlfWorld", choices = ['Alfworld'], help="Environment")
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--exp-name", type=str, default="small", help="name of the experiment")
parser.add_argument("--quit-ends-game", action="store_true", help="Should quitting end the game or be considered an invalid action?")
parser.add_argument("--quit-hint", action = 'store_true')

def initialize_logging(run_id, args):
    "Creates a dictionary which is pickled at the end."
    stored_run = dict()
    stored_run['run_id'] = run_id
    stored_run['args'] = args
    stored_run['task_type'] = []
    stored_run['task_name'] = []
    stored_run['task_success'] = []
    stored_run['task_length'] = []
    stored_run['trajectory'] = []
    return stored_run

def set_up_alfworld_agent(agent, llm):
    llm = LLM.get_llm(llm)

    if agent == 'ReAct':
        prompt_file = './Alfworld/prompts/alfworld_3prompts.json'
        
        TASK_TYPES = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
        } 
        
        return ReAct(llm, TASK_TYPES, prompt_file)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    
    args = parser.parse_args()
    RUN_ID = args.exp_name + '_' + args.llm + '_' + args.agent
    stored_run = initialize_logging(RUN_ID, args)

    if args.env == 'AlfWorld':
        agent = set_up_alfworld_agent(args.agent, args.llm)
        env = Env.set_up_alfworld_env(args.exp_name + '_config.yaml')
    else: 
        raise NotImplementedError
    
    if args.quit_ends_game:
        env = Env.QuitEnv(env)
        if args.quit_hint:
            quit_hint = 'If you believe you will fail the task, you can say "think: quit". Do not quit unless you are confident you cannot complete the task.'
        else: 
            quit_hint = ''
    
    # Evaluate on N runs, storing results in `stored_run`
    evaluator = Evaluator(agent, env, args.N, stored_run) 
    n_evaluated = evaluator(quit_hint, to_print = False)
    if n_evaluated == 0:
        exit(1)

    # pickle stored run
    write_folder = f'./experiments/{args.env}/{args.agent}/{args.llm}/{args.exp_name}/n={args.N}'
    file_name = f'/run_{str(datetime.now())}'
    utils.save_run(write_folder, file_name, stored_run)
    
    exit(0)