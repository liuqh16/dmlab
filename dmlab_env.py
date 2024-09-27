# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to facilitate running DMLab env.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gym
import numpy as np
import deepmind_lab
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


def create_env_settings(level_name, homepath='', width=64, height=64, seed=0,
                        main_observation='DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'):
    """Creates environment settings."""
    env_settings = {
        'seed':
            seed,
        # See available levels:
        # https://github.com/deepmind/lab/tree/master/game_scripts/levels
        'levelName':
            level_name,
        'width':
            width,
        'height':
            height,
        # Documentation about the available observations:
        # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
        'observationFormat': [
            main_observation,
            'DEBUG.POS.TRANS',
            'MAP_FRAME_NUMBER',
            'DEBUG.MAZE.LAYOUT',
            'DEBUG.POS.ROT',
            'DEBUG.PLAYERS.VELOCITY',
        ],
        'homepath':
            homepath,
        'logLevel':
            -1,
    }
    return env_settings


# A set of allowed actions.
DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)

DEFAULT_ACTION_SET_WITH_IDLE = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),  # Fire.
    (0, 0, 0, 0, 0, 0, 0),  # Idle.
)

# Default set without "Fire".
DEFAULT_ACTION_SET_WITHOUT_FIRE = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
)

# A small action set.
ACTION_SET_SMALL = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
)


# Another set of actions with idle.
ACTION_SET_WITH_IDLE = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (0, 0, 0, 0, 0, 0, 0),    # Idle.
)


ACTION_SET_SMALL_WITH_BACK = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
)


class DMLabWrapper(gym.Env):
    """A wrapper around DMLab environment to make it compatible with OpenAI Baseline's training."""

    def __init__(self, platform, args,
                 action_set=DEFAULT_ACTION_SET,
                 main_observation='DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE',
                 action_repeat=4):
        """Creates a DMLabWrapper.

        Args:
        platform: Typically 'dmlab'.
        args: The environment settings.
        action_set: The set of discrete actions.
        main_observation: The observation returned at every time step.
        action_repeat: Maximum number of times to repeat an action. This can be less at the end of an episode.
        """
        homepath = args.pop('homepath')
        level_name = args.pop('levelName')
        observation_format = args.pop('observationFormat')
        seed = args.pop('seed')
        string_args = {key: str(value) for key, value in args.items()}
        if homepath:
            deepmind_lab.set_runfiles_path(os.path.join(homepath))
        self._env = deepmind_lab.Lab(level_name, observation_format, string_args)

        self._random_state = np.random.RandomState(seed=seed)
        # self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))

        self._action_set = action_set
        self._action_repeat = action_repeat
        self.width = args['width']
        self.height = args['height']

        self._main_observation = main_observation
        self._transform_observation = lambda x: x
        if main_observation == 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE':
            # This observation format is (RGB, height, width).
            # Convert it to (height, width, RGB).
            self._transform_observation = lambda x: np.moveaxis(x, 0, -1)

        # Build a list of all the possible actions.
        self._action_list = []
        for action in action_set:
            self._action_list.append(np.array(action, dtype=np.intc))

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._action_set))

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

    def reset(self):
        self._current_noise = np.zeros((int(self.observation_space.shape[0]/2),
                                        int(self.observation_space.shape[1]/2),
                                        self.observation_space.shape[2]))
        self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
        time_step = self._env.observations()
        main_observation = self._transform_observation(time_step[self._main_observation])
        return main_observation

    def close(self):
        self._env.close()

    def step(self, action):
        """Performs one step in the environment.

        Args:
        action: which action to take
        Returns:
        A tuple (observation, reward, done, metadata)
        """
        reward = self._env.step(self._action_list[action], num_steps=self._action_repeat)
        done = np.array(not self._env.is_running())
        if not done:
            time_step = self._env.observations()
            main_observation = self._transform_observation(time_step[self._main_observation])
        else:
            main_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        # if done:
        #     self.reset()
        # time_step = self._env.observations()
        # main_observation = self._transform_observation(time_step[self._main_observation])
        # metadata = {
        #     'position': time_step['DEBUG.POS.TRANS'],
        #     'frame_num': time_step['MAP_FRAME_NUMBER'],
        #     'maze_layout': time_step['DEBUG.MAZE.LAYOUT'],
        #     'rotation': time_step['DEBUG.POS.ROT'],
        #     'velocity': time_step['DEBUG.PLAYERS.VELOCITY'],
        # }
        return (main_observation, reward, done, {})


def make_dmlab_env(dmlab_level_name, num_env=8, seed=None, small_action_set=False):
    """Creates a DMLab environment."""
    def make_env(seed):
        def _thunk():
            env_settings = create_env_settings(dmlab_level_name, seed=seed)
            action_set = ACTION_SET_SMALL if small_action_set else DEFAULT_ACTION_SET
            env = DMLabWrapper('dmlab', env_settings, action_set=action_set)
            return env
        return _thunk

    seed = seed or np.random.randint(0, 2**31 - 1)
    return SubprocVecEnv([make_env(seed + i) for i in range(num_env)])


if __name__ == "__main__":
    venv = make_dmlab_env('contributed/dmlab30/explore_goal_locations_large', num_env=8)
    obs = venv.reset()
    print(obs.shape)
    action = [venv.action_space.sample() for _ in range(8)]
    obs, reward, done, info = venv.step(action)
    print(obs.shape, reward, done, info)
