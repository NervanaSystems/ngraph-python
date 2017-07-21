from __future__ import print_function

import gym
import ngraph as ng
from ngraph.examples.dqn import dqn
from ngraph.examples.dqn import rl_loop
from ngraph.frontends import neon


def model(action_axes):
    return neon.Sequential([
        neon.Affine(
            nout=20,
            weight_init=neon.XavierInit(),
            activation=neon.Tanh(),
            batch_norm=True,
        ),
        neon.Affine(
            nout=20,
            weight_init=neon.XavierInit(),
            activation=neon.Tanh(),
            batch_norm=True,
        ),
        neon.Affine(
            nout=20,
            weight_init=neon.XavierInit(),
            activation=neon.Tanh(),
            batch_norm=True,
        ),
        neon.Affine(
            axes=action_axes,
            weight_init=neon.XavierInit(),
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
    ])


def baselines_model(action_axes):
    return neon.Sequential([
        neon.Affine(
            nout=64,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
            batch_norm=False,
        ),
        neon.Affine(
            axes=action_axes,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=None,
        ),
    ])


def main():
    # initialize gym environment
    environment = gym.make('CartPole-v0')

    state_axes = ng.make_axes([
        ng.make_axis(environment.observation_space.shape[0], name='width')
    ])

    agent = dqn.Agent(
        state_axes,
        environment.action_space,
        model=baselines_model,
        epsilon=dqn.linear_generator(start=1.0, end=0.02, steps=10000),
        learning_rate=1e-3,
        gamma=1.0,
        memory=dqn.Memory(maxlen=50000),
        learning_starts=1000,
    )

    rl_loop.rl_loop_train(environment, agent, episodes=1000)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)
    print(total_reward)


if __name__ == "__main__":
    main()
