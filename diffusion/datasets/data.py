import pdb 
import gym 
import d4rl
import numpy as np 

def load_environment(name: str):
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name 
    return env 

def get_episodes(dataset):
    '''
    Given the d4rl dataset, return list of episodes represented as a dictionary that contain obs, action, reward, terminals\
        timeouts arrays
    '''
    episode_start_index = 0
    episodes_data = [] #store the episodes as dictionaries with key (str): value (array)

    N = dataset['rewards'].shape[0] #N represents the lengths of the arrays of dataset
    for i in range(N):
        is_done = bool(dataset['terminals'][i])
        is_timeout = bool(dataset['timeouts'][i])
        if is_done or is_timeout:
            episode_stop_index = i
            episode = {}
            for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
                episode[key] = np.array(dataset[key][episode_start_index: episode_stop_index + 1])

            episodes_data.append(episode)    
            episode_start_index = i+1 


    return episodes_data

                    
def process_dataset(env_name):
    '''
    Returns a list of episodes (dict) that contain obs, action, reward, terminals\
        timeouts arrays
    '''
    env = load_environment(env_name)
    #TODO: if bad, add fix timeouts and scale rewards as in janner
    dataset = env.get_dataset()
    return get_episodes(dataset)





