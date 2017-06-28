from collections import deque

import cv2
import gym
import numpy as np


class ReshapeWrapper(gym.Wrapper):
    """
    Reshape the observation provided by open ai gym atari environment to match
    the deepmind dqn paper.
    """

    def __init__(self, environment):
        super(ReshapeWrapper, self).__init__(environment)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84))

    def _modify_observation(self, observation):
        # convert color to grayscale using luma component
        observation = (
            observation[:, :, 0] * 0.299 + observation[:, :, 1] * 0.587 +
            observation[:, :, 2] * 0.114
        )

        observation = cv2.resize(
            observation, (84, 110), interpolation=cv2.INTER_AREA
        )
        observation = observation[18:102, :]
        assert observation.shape == (84, 84)

        # convert to values between 0 and 1
        observation = np.array(observation, dtype=np.uint8)

        return observation

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self._modify_observation(observation)

        return observation, reward, done, info

    def _reset(self):
        return self._modify_observation(self.env.reset())


class ClipRewardWrapper(gym.Wrapper):
    """
    wraps an environment so that the reward is always either -1, 0 or 1
    """

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)

        # clip reward to -1, 0, or 1
        if reward > 0:
            reward = 1.0
        elif reward < 0:
            reward = -1.0
        else:
            reward = 0.0

        return observation, reward, done, info


class LazyStack(object):
    """
    A lazy version of np.stack which avoids copying the memory until it is
    needed.
    """

    def __init__(self, history, axis=None):
        self.history = history
        self.axis = axis

    def __array__(self, dtype=None):
        array = np.stack(self.history, axis=self.axis)
        if dtype is not None:
            array = array.astype(dtype)
        return array


class RepeatWrapper(gym.Wrapper):
    """
    Send multiple steps of observations to agent at each step
    """

    def __init__(self, env, frames=4):
        super(RepeatWrapper, self).__init__(env)
        self.frames = frames

        # todo: this shouldn't always be a box, low and high aren't
        #       always 0 and 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(frames, ) + self.observation_space.shape,
        )

    def _reset(self):
        self.history = deque([self.env.reset()], maxlen=self.frames)

        # take random actions to start and fill frame buffer
        for _ in range(self.frames - 1):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            self.history.append(observation)
            assert done is not True

        return self._get_observation()

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.history.append(observation)

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return LazyStack(self.history, axis=0)


class TerminateOnEndOfLifeWrapper(gym.Wrapper):
    """
    treat the end-of-life the same as termination.
    """

    def __init__(self, env):
        super(TerminateOnEndOfLifeWrapper, self).__init__(env)

        self.last_lives = 0
        self.needs_reset = True

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.needs_reset = done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.last_lives:
            done = True

        self.last_lives = lives

        return observation, reward, done, info

    def _reset(self):
        # only reset the parent environment if the parent environment triggered
        # the termination. if we are getting reset because a life was lost,
        # just take a normal step and pretend it was a reset.
        if self.needs_reset:
            self.needs_reset = False
            observation = self.env.reset()
        else:
            observation, _, self.needs_reset, _ = self.env.step(0)
            assert self.needs_reset is not True

        self.last_lives = self.env.unwrapped.ale.lives()

        return observation


class DimShuffleWrapper(gym.Wrapper):
    """
    Reshape the observation provided by open ai gym atari environment to match
    the deepmind dqn paper.
    """

    def __init__(self, environment):
        super(DimShuffleWrapper, self).__init__(environment)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4, 84, 84)
        )

    def _modify_observation(self, observation):
        observation = np.asarray(observation)

        observation = observation.transpose((2, 0, 1))
        observation = np.ascontiguousarray(observation)

        return observation

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self._modify_observation(observation)

        return observation, reward, done, info

    def _reset(self):
        return self._modify_observation(self.env.reset())