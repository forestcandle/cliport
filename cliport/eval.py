"""Ravens main training script."""

import os
import pickle
import json

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.utils import command_strings
from cliport.environments.environment import Environment
from optparse import OptionParser
import pandas as pd

# parser = OptionParser()
# # Random seed
# parser.add_option("--seed", type="int", dest="seed", default=0)
# # Description of target item to be used in language prompt
# parser.add_option("--target_item_desc", dest="target_item_desc", default="cube")

@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    # options, args = parser.parse_args()

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    start_seed = vcfg['seed']
    target_item_desc = vcfg['target_item_desc']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    elif mode is not 'test':
        # TODO HACK test mode disabled for race experiments, make code work properly
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}")

            record = vcfg['record']['save_video']
            n_demos = vcfg['n_demos']
            # HACK TODO clean up command string iteration, re-enable other tasks
            command_strings = command_strings.build_command_strings('reduced')
            num_command_strings = len(command_strings)
            n_demos_per_command = n_demos
            n_demos = n_demos * num_command_strings
            folds = vcfg['folds']
            fold = vcfg['fold']
            command_string_min = 0
            command_string_max = max(num_command_strings, 1)
            if folds > 0:
                num_strings_in_fold = np.ceil(float(num_command_strings) / float(folds))
                command_string_min = num_strings_in_fold * fold
                command_string_max = min(num_strings_in_fold * (fold + 1), num_command_strings)

            # Run testing and save total rewards with last transition info.
            for j in range(command_string_min, command_string_max):
                for i in range(0, n_demos):
                    k = (j+1) * (i+1) + i
                    print(f'Test: total {k}/{n_demos*num_command_strings} demos {i + 1}/{n_demos} commands: {j + 1}/{n_demos}')
                    if mode is not 'test':
                        episode = k
                        seed = start_seed + i
                    else:
                        episode, seed = ds.load(i)
                    np.random.seed(seed)
                    current_command_string = command_strings[j]
                    pd_save_path = save_json[:save_json.rindex("/")]
                    pd_save_path = os.path.join(pd_save_path, f"run_csv_seed-{seed}_run-{i}_desc-{current_command_string}.csv")
                    if os.path.exists(pd_save_path):
                        # already ran this experiment, so skip to the next one
                        continue

                    goal = episode[-1]
                    total_reward = 0
                    object_infos={}

                    # set task
                    if 'multi' in dataset_type:
                        task_name = ds.get_curr_task()
                        task = tasks.names[task_name](current_command_string)
                        print(f'Evaluating on {task_name}')
                    else:
                        task_name = vcfg['eval_task']
                        task = tasks.names[task_name]()

                    task.mode = mode
                    num_command_strings = len(task.command_strings)
                    if num_command_strings:
                        task.lang_template = task.command_strings[j]

                    # env.seed(np.random.randint(2**32-1))
                    env.seed(seed)
                    env.set_task(task)
                    obs = env.reset()
                    info = env.info
                    reward = 0

                    for obj_id in task.object_log_info:
                        object_infos[obj_id]=[]
                        object_infos[obj_id].append(task.object_log_info[obj_id])

                    # Start recording video (NOTE: super slow)
                    if record:
                        video_name = f'{task_name}-{i+1:06d}'
                        if 'multi' in vcfg['model_task']:
                            video_name = f"{vcfg['model_task']}-{video_name}"
                        env.start_rec(video_name)

                    for _ in range(task.max_steps):
                        act = agent.act(obs, info, goal)
                        lang_goal = info['lang_goal']
                        print(f'Lang Goal: {lang_goal}')
                        obs, reward, done, info = env.step(act)

                        for obj_id in info['pose']:
                            object_infos[obj_id].append((info['pose'][obj_id], info['placed'][obj_id]))

                        total_reward += reward
                        print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                        if done:
                            break

                    df = pd.DataFrame(data=object_infos)
                    df.to_csv(pd_save_path)

                    results.append((total_reward, info))
                    mean_reward = np.mean([r for r, i in results])
                    print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}')

                    # End recording video
                    if record:
                        env.end_rec()

                all_results[ckpt] = {
                    'episodes': results,
                    'mean_reward': mean_reward,
                }

        # Save results in a json file.
        if vcfg['save_results']:

            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, 'w') as f:
                json.dump(all_results, f, indent=4)


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        print('looking for checkpoints in: ' + vcfg['model_path'] ' while excluding existing results: ' + str(existing_results))
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        print('looking for checkpoints in: ' + vcfg['model_path'])
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    if not ckpts_to_eval:
        raise FileNotFoundError('Could not find any model checkpoint files to load, check the folders specified and the eval.py function list_ckpts_to_eval().')

    return ckpts_to_eval


if __name__ == '__main__':
    main()
