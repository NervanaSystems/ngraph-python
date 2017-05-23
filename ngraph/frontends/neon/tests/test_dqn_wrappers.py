import gym

import dqn_space_invaders


class MockEnvironment(object):
    """environment which generates very simple observations for testing"""

    def __init__(self, observations):
        super(MockEnvironment, self).__init__()
        self.observations = iter(observations)
        self.metadata = {}
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = None
        self.reward_range = None

    def reset(self):
        return self.observations.next()

    def _step(self, action):
        return self.observations.next(), 0, False, None


def test_repeat_wrapper_reset():
    environment = MockEnvironment(range(10))
    environment = dqn_space_invaders.RepeatWrapper(frames=2)(environment)

    assert list(environment._reset()) == [0, 1]


def test_repeat_wrapper_step():
    environment = MockEnvironment(range(10))
    environment = dqn_space_invaders.RepeatWrapper(frames=2)(environment)
    environment.reset()

    assert list(environment.step(1)[0]) == [1, 2]
    assert list(environment.step(1)[0]) == [2, 3]
