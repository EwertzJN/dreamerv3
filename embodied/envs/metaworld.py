import json
import random

import embodied
import numpy as np

import gymnasium
from gymnasium.core import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

DEFAULT_CAMERA_CONFIG = {
  "distance": 1.25,
  "azimuth": 145,
  "elevation": -40.0,
  "lookat": np.array([-0.05, 0.75, 0.0]),
}

DEFAULT_SIZE = 64


class CameraWrapper(gymnasium.Wrapper):
  def __init__(self, env: Env, seed: int):
    super().__init__(env)

    self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
    self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
    self.unwrapped.mujoco_renderer = MujocoRenderer(env.model, env.data, DEFAULT_CAMERA_CONFIG, DEFAULT_SIZE,
                                                    DEFAULT_SIZE)

    # Hack: enable random reset
    self.unwrapped._freeze_rand_vec = False
    self.unwrapped.seed(seed)

  def reset(self):
    obs, info = super().reset()
    return obs, info

  def step(self, action):
    next_obs, reward, done, truncate, info = self.env.step(action)
    return next_obs, reward, done, truncate, info


def setup_metaworld_env(task_name: str, seed: int):
  env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
  env = CameraWrapper(env_cls(render_mode="rgb_array"), seed)
  return env


class MetaWorld(embodied.Env):

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None, repeat=10):
    self.action_repeat = repeat
    self.observation_size = size
    self.seed = random.sample(range(1, 1000000), 1)[0] if seed is None else seed
    self._logs = logs
    self._logdir = logdir and embodied.Path(logdir)
    self._logdir and self._logdir.mkdir()

    if task == "button_press":
      self._env = setup_metaworld_env("button-press-v2-goal-observable", self.seed)
    elif task == "hammer":
      self._env = setup_metaworld_env("hammer-v2-goal-observable", self.seed)
    elif task == "box_close":
      self._env = setup_metaworld_env("box-close-v2-goal-observable", self.seed)
    elif task == "drawer_close":
      self._env = setup_metaworld_env("drawer-close-v2-goal-observable", self.seed)
    else:
      raise ValueError(f"Unknown task: {task}")

    self.current_success = 0

    self._episode = 0
    self._length = None
    self._reward = None
    self._success = False
    self._done = True

  @property
  def obs_space(self):
    spaces = {
      'image': embodied.Space(np.uint8, (64, 64, 3)),
      'reward': embodied.Space(np.float32),
      'is_first': embodied.Space(bool),
      'is_last': embodied.Space(bool),
      'is_terminal': embodied.Space(bool),
    }
    if self._logs:
      spaces.update({
          f'is_success': embodied.Space(bool)
      })
    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.float32, self._env.action_space.shape[0], float(self._env.action_space.low[0]), float(self._env.action_space.high[0])),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self.current_success = 0
      self._done = False
      _ = self._env.reset()
      image = self._env.render()
      return self._obs(image, 0.0, is_first=True)
    for k in range(self.action_repeat):
      _, reward, _, self._done, info = self._env.step(action['action'])
      self.current_success = min(info['success'] + self.current_success, 1.0)
      self._length += 1
      if self._done and self._logdir:
        self._write_stats(self._length, self._reward)
      if self._done:
        break
    self._reward += reward
    image = self._env.render()
    return self._obs(image, reward, is_last=self._done)

  def _obs(self, image, reward, is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    if self._logs:
      log_achievements = {
          'is_success': bool(self.current_success)}
      obs.update({k: v for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': reward,
        'is_success': bool(self.current_success),
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines)
    print(f'Wrote stats: {filename}')

  def render(self):
    return self._env.render()

  def close(self):
    self._env.close()

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_range(self):
    return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])
