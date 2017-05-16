import gym
import numpy as np
import cv2
import simple_environments
from ngraph.frontends.neon import dqn, rl_loop
from ngraph.frontends import neon
import ngraph as ng

# factory = ng.transformers.make_transformer_factory('gpu')
# mg.transformers.set_transformer_factory(factory)


def model(action_axes):
    print(action_axes.length)
    return neon.Sequential([
        neon.Convolution(
            (8, 8, 32),
            neon.XavierInit(),
            strides=4,
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
        neon.Convolution(
            (4, 4, 64),
            neon.XavierInit(),
            strides=2,
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
        neon.Convolution(
            (3, 3, 64),
            neon.XavierInit(),
            strides=1,
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
        neon.Affine(
            nout=512,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
        neon.Affine(
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
            batch_norm=True,
            axes=(action_axes, )
        ),
    ])

class ReshapeWrapper(gym.Wrapper):
    """docstring for ReshapeWrapper."""
    def __init__(self, environment):
        super(ReshapeWrapper, self).__init__(environment)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84))

    def _modify_observation(self, observation):
        # convert color to grayscale
        observation = np.mean(observation, axis=2)
        # resize image to 84, 84
        observation = cv2.resize(observation, (84, 84))
        # convert to values between 0 and 1
        observation = observation / 256.

        return observation

    def _step(self, action):
        observation, reward, done, info = self.env._step(action)

        observation = self._modify_observation(observation)

        return observation, reward, done, info

    def _reset(self):
        return self._modify_observation(self.env._reset())


def main():
    # todo: total_reward isn't always greater than 95 even with a working implementation
    # environment = gym.make('SpaceInvaders-v0')
    environment = gym.make('Pong-v0')
    environment = ReshapeWrapper(environment)

    print(environment.observation_space)

    # todo: perhaps these should be defined in the environment itself
    state_axes = ng.make_axes([
        # ng.make_axis(environment.observation_space.shape[2], name='feature'),
        ng.make_axis(environment.observation_space.shape[0], name='width'),
        ng.make_axis(environment.observation_space.shape[1], name='height'),
    ])

    agent = dqn.Agent(
        state_axes,
        environment.action_space,
        model=model
    )

    rl_loop.rl_loop(environment, agent, episodes=20000)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)

    assert total_reward >= 95


if __name__ == "__main__":
    main()
