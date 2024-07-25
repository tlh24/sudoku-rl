'''
Visualizes the diffusion plan by saving video of observations 
From diffuser
'''
import os
import numpy as np
import skvideo.io
from tqdm import tqdm 
import imageio
import torch 
import warnings
import mujoco_py as mjc
from PIL import Image 

## video generation ##
def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(filename, video_frames, fps=60, video_format='mp4'):
    assert fps == int(fps), fps

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )

## helper functions ##
def get_image_mask(img):
    # Returns mask that is True for non-all-white pixels, False for all white pixels
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def create_background_mask(image, tolerance=80):
    '''
    Returns a boolean array of same shape as image; pixel is true iff non maze2d background
    '''
    img_array = np.array(image)
    
    # Define the background colors
    bg_color1 = np.array([25, 51, 76])  # Dark blue
    bg_color2 = np.array([51, 76, 102])  # Lighter blue
    
    diff1 = np.sum(np.abs(img_array - bg_color1), axis=-1)
    diff2 = np.sum(np.abs(img_array - bg_color2), axis=-1)
    
    # Create a mask where pixels are true if NOT close to either background color
    background_mask = (diff1 > tolerance) & (diff2 > tolerance)
    return background_mask


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])


## renderer ##
class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, dim=(1024, 256), qvel=True, render_kwargs=None,**kwargs):
        '''
        Renders a path and returns a collated image of the observations
            For our purposes, partial should remain false

        samples: (H, obs_dim )
        
        '''
        assert partial == False 
        if partial:
            samples = self.pad_observations(samples)
            partial = False
        
        if render_kwargs is None:
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 10,
                'lookat': [5, 5, 0],
                'elevation': -90.0
            }

        sample_images = self._renders(samples, partial=partial,dim=dim, qvel=qvel,render_kwargs=render_kwargs, **kwargs)
        # (H, img_dims)

        composite = np.ones_like(sample_images[0]) * 255
        for img in sample_images:
            mask = create_background_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 5, 0],
            'elevation': -90.0
        }
        images = []
        for path in paths:
            ## path is [ H x obs_dim ]
            path = atmost_2d(path)
            # get a composite image representing all the obs in the path
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images


def show_diffusion(renderer, observations, n_repeat=100, substep=100, filename='diffusion.mp4', savefolder='videos/'):
    '''
        Saves a video which is a composition of visualization of generated plans 
        observations : [ n_diffusion_steps x batch_size x horizon x observation_dim ]
    '''
    savepath = os.path.join(savefolder, filename)
    subsampled = observations[::substep]

    images = []
    for t in tqdm(range(len(subsampled))):
        diffusion_plan = subsampled[t]
        img = renderer.composite(None, diffusion_plan)
        images.append(img)
    images = np.stack(images, axis=0)

    ## pause at the end of video
    images = np.concatenate([
        images,
        images[-1:].repeat(n_repeat, axis=0)
    ], axis=0)

    save_video(savepath, images)

def show_plan_over_time(renderer, observations, savefolder='images/'):
    '''
    observations: [n_env_steps, horizon, obs_dim]
    '''
    # create save folder if doesn't exist
    _make_dir(os.path.join(savefolder, "fake_file.png"))

    for step in range(0, len(observations)):
        diffusion_plan = observations[step]
        diffusion_plan = np.expand_dims(diffusion_plan, axis=0) #(1,horizon,obs_dim)
        np_img = renderer.composite(None, diffusion_plan)
        img = Image.fromarray(np_img)
        filename = f'envstep{step + 1}_plan.png'
        savepath = os.path.join(savefolder, filename)
        img.save(savepath)
    
    return 








