import random
import numpy as np
import gym
from gym.envs.registration import registry, register, make, spec
from gym import spaces


class ConstantEnv(gym.Env):
    """an environment where a single discrete action is always the best no matter the state"""

    def __init__(self):
        super(ConstantEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Discrete(2)

    def _reset(self):
        self.steps = 0
        return np.zeros((2, ))

    def _get_observation(self):
        return np.zeros((2, ))

    def _terminate(self):
        return self.steps >= 100

    def _reward(self, action):
        return int(action == 1)

    def _step(self, action):
        self.steps += 1
        return (
            self._get_observation(), self._reward(action), self._terminate(),
            {},
        )


class RandomInputConstantGoalEnv(gym.Env):
    """docstring for RandomInputConstantGoalEnv."""

    def __init__(self):
        super(RandomInputConstantGoalEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Discrete(2)

    def _reset(self):
        self.steps = 0
        return self._get_observation()

    def _get_observation(self):
        return random.choice([
            np.array([1, 0]),
            np.array([0, 1]),
        ])

    def _terminate(self):
        return self.steps >= 100

    def _reward(self, action):
        return int(action == 1)

    def _step(self, action):
        self.steps += 1
        return (
            self._get_observation(), self._reward(action), self._terminate(),
            {},
        )


class DependentEnv(gym.Env):
    """docstring for DependentEnv."""

    def __init__(self):
        super(DependentEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Discrete(2)

    def _reset(self):
        self.steps = 0
        return self._get_observation()

    def _get_observation(self):
        self.last_observation = random.choice([
            np.array([1, 0]),
            np.array([0, 1]),
        ])
        return self.last_observation

    def _terminate(self, reward):
        return not reward
        return self.steps >= 100

    def _reward(self, observation, action):
        return int(np.argmax(observation) == action)

    def _step(self, action):
        self.steps += 1
        reward = self._reward(self.last_observation, action)
        observation = self._get_observation()
        return (observation, reward, self._terminate(reward), {}, )


register(
    id='ConstantEnv-v0',
    entry_point='simple_environments:ConstantEnv',
)

register(
    id='RandomInputConstantGoalEnv-v0',
    entry_point='simple_environments:RandomInputConstantGoalEnv',
)
register(
    id='DependentEnv-v0',
    entry_point='simple_environments:DependentEnv',
)
