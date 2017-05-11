import gym
import simple_environments
import rl_loop
from dqn import Agent
from ngraph.frontends import neon
import ngraph

factory = ngraph.transformers.make_transformer_factory('gpu')
ngraph.transformers.set_transformer_factory(factory)


def small_model(action_axes):
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


def main():
    # todo: total_reward isn't always greater than 95 even with a working implementation
    environment = gym.make('SpaceInvaders-v0')
    agent = Agent(
        environment.observation_space,
        environment.action_space,
        model=small_model
    )

    rl_loop.rl_loop(environment, agent, episodes=20)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)

    assert total_reward >= 95


if __name__ == "__main__":
    main()
