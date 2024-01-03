'''
Collect Trajectories to be scored for RLAIF

Rough ideas

		want it to be diverse It hink. doesn't need to be expert only
		motif does caption plus score
			would want to store like episode, step, actions, rewards, obs, info, done, captions, other relevant info (like how it's gathered, random seed etc)
			could do more important stuff later and just do caption for now

Wish list
async, paralellized, efficient computing and saving
works with RLease
classes in efficient way, handles different types of envs
saves to AWS

later
argue in base_dir with more args/ os specific / cloud specific
'''

import time
import gymnasium as gym
import rlease_utils
import argparse
from rlease_agent import RLeaseAgentRandom
from rlease_trajector_stats import get_trajectory_stats


def main(args):
    # read args
    # env name, num episodes, other settings
    args_dict = rlease_utils.read_yaml_file(args.yaml_settings_path)

    # constants
    time_int = int(time.time())

    # create base directory
    base_dir = '/RLAIF_experiment/base_save_dir_{}/'.format(time_int)
    rlease_utils.make_base_dir(base_dir)

    # save experiment settings
    rlease_utils.save_pickle_file(base_dir + "experiment_settings.pkl", args_dict)

    is_success = run_and_save_trajectories(args_dict, base_dir)

    print("__ Experiment finished {} __".format(is_success), base_dir)


def run_and_save_trajectories(args_dict, base_dir):
    '''
    Set up env
    Collect env trajectories
    Save trajectories
    Save overall stats and other useful logging
    '''
    is_success = False
    try:
        # start up env
        env = create_env(args_dict)

        # get agent
        agent = RLeaseAgentRandom()

        # # run and save trajectories
        # stats_dict = collect_and_save_trajectories(env, args_dict, base_dir)

        # # save stats summary
        # save_experiment_settings(stats_dict, base_dir)

        df_eval, action_dict_counter, agent_logging_dict = get_trajectory_stats(
            agent, env, args_dict, base_dir, is_agent_logging=False)

        rlease_utils.save_pickle_file(base_dir + "df_eval.pkl", df_eval)
        is_success = True
    except Exception as e:
        print("Error: run and save trajectories failed ", str(e))
    finally:
        return is_success


def create_env(args_dict):
    env = gym.make(args_dict.get('env', 'Blackjack-v1'))

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_settings_path', 
        help='File path for YAML settings for running experiment', required=True)

    args = parser.parse_args()
    main(args)
