import json
import random

import numpy as np

import embodied
import visual_block_builder


class RobotEnv:
  def __init__(self, env, seed, action_repeat, observation_size):
    import logging
    import gym
    gym.logger.set_level(logging.ERROR)  # Ignore warnings from Gym logger
    self._env = gym.make(env)
    self._env.seed(seed)
    self.action_repeat = action_repeat
    self.observation_size = observation_size

  def reset(self):
    self.t = 0  # Reset internal timer
    _ = self._env.reset()
    return self._env.render(mode='rgb_array', size=self.observation_size)

  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state, reward, done, _ = self._env.step(action)
      self.t += 1  # Increment internal timer
      if done:
        break
    observation = self._env.render(mode='rgb_array', size=self.observation_size)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_range(self):
    return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])


class MultiRobotEnv(embodied.Env):

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None, repeat=2):
    self.action_repeat = repeat
    self.observation_size = size
    self.seed = seed
    self._logs = logs
    self._logdir = logdir and embodied.Path(logdir)
    self._logdir and self._logdir.mkdir()

    self.envs = [task.split('_') for _ in range(int(task.split('_')[1].split('to')[0]), int(task.split('_')[1].split('to')[1]) + 1)]
    for i in range(int(task.split('_')[1].split('to')[0]), int(task.split('_')[1].split('to')[1]) + 1):
      index = i - int(task.split('_')[1].split('to')[0])

      self.envs[index][2] = str(i) + self.envs[index][2]
      del self.envs[index][1]
      self.envs[index] = '_'.join(self.envs[index])

    self._env = RobotEnv(self.envs[0], seed, self.action_repeat, self.observation_size)._env
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
      self._done = False
      env = random.choice(self.envs)
      self._env = RobotEnv(env, self.seed, self.action_repeat, self.observation_size)._env
      _ = self._env.reset()
      image = self._env.render(mode='rgb_array', size=self.observation_size)
      return self._obs(image, 0.0, is_first=True)
    for k in range(self.action_repeat):
      state, reward, self._done, _ = self._env.step(action['action'])
      self._reward += reward
      self._length += 1
      if self._done and self._logdir:
        self._write_stats(self._length, self._reward)
      if self._done:
        break
    image = self._env.render(mode='rgb_array', size=self.observation_size)
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
          'is_success': self.success()}
      obs.update({k: v for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        'is_success': self.success(),
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines)
    print(f'Wrote stats: {filename}')

  def success(self):
    return bool(self._env.env.success())

  def dist(self):
    return self._env.env.dist()

  def render(self):
    return self._env.render(mode='rgb_array', size=self.observation_size)

  def close(self):
    self._env.close()

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_range(self):
    return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])
