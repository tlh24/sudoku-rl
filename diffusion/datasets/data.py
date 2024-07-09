import pdb 
import gym 
import d4rl
import numpy as np 
import torch

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()
class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

def load_environment(name: str):
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name 
    return env 






class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, env_name='maze2d-large-v1', horizon=64):
        self.env_name = env_name
        env = load_environment(self.env_name)
        self.dataset = env.get_dataset()
        self.normalizers = self.get_normalizers()
        self.episodes = self.get_episodes()
        self.horizon = horizon 
        self.indices = self.make_indices()
        
    def get_normalizers(self):
        '''
        Returns a dictionary where key is 'observations', 'actions
            and value is LimitsNormalizer
        '''
        normalizers = {}
        for key in ['observations', 'actions']:
            normalizers[key] =  LimitsNormalizer(self.dataset[key])
        return normalizers

    def get_episodes(self):
        '''
        Given the d4rl dataset, return list of episodes represented as a dictionary that contain obs, action, reward, terminals\
            timeouts arrays
        '''
        dataset = self.datset
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
                    if key in ['observations', 'actions']:
                        episode[f'norm_{key}'] = self.normalizers[key].normalize(episode[key])

                episodes_data.append(episode)    
                episode_start_index = i+1 
        return episodes_data

    def make_indices(self):
        '''
        Returns a list of (episode_idx, start_idx, end_idx) which is every possible
            horizon long subsequence; end_idx is non-inclusive 
        '''
        num_skipped = 0
        indices = []
        for episode_idx in range(len(self.episodes)):
            episode = self.episodes[episode_idx]
            if len(episode['observations']) < self.horizon:
                num_skipped += 1
                continue
            for start_idx in range(len(episode['observations']) - self.horizon + 1):
                indices.append((episode_idx, start_idx, start_idx + self.horizon))
        print(f"Skipped {num_skipped} episodes shorter than horizon {self.horizon}")
        return indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, start_idx, end_idx = self.indices[idx]
        episode = self.episodes[episode_idx]
        observation_arr = episode['norm_observations'][start_idx:end_idx]
        action_arr = episode['norm_actions'][start_idx:end_idx]
        #trajectories = np.concatenate([action_arr, observation_arr], axis=-1)
        return observation_arr, action_arr

    


