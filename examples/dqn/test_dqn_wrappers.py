import examples.dqn.gym_wrapper
import numpy as np
import gym


class MockEnvironment(object):
    """environment which generates very simple observations for testing"""

    def __init__(self, observations):
        super(MockEnvironment, self).__init__()
        self.observations = iter(observations)
        self.metadata = {}
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 2))
        self.reward_range = None

    def reset(self):
        return next(self.observations)

    def step(self, action):
        return next(self.observations), 0, False, None


def test_repeat_wrapper_reset():
    environment = MockEnvironment(list(range(10)))
    environment = examples.dqn.gym_wrapper.RepeatWrapper(environment, frames=2)

    assert list(np.array(environment._reset())) == [0, 1]


def test_repeat_wrapper_step():
    environment = MockEnvironment(list(range(10)))
    environment = examples.dqn.gym_wrapper.RepeatWrapper(environment, frames=2)
    environment.reset()

    assert (environment.step(1)[0] == np.array([1, 2])).all()
    assert (environment.step(1)[0] == np.array([2, 3])).all()
