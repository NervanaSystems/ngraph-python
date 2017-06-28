import gym
import simple_environments  # NOQA
from ngraph.frontends.neon import dqn, rl_loop
from ngraph.frontends import neon
import numpy as np


def model(action_axes):
    return neon.Sequential([
        neon.Affine(
            nout=10,
            weight_init=neon.GlorotInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Tanh(),
        ),
        neon.Affine(
            weight_init=neon.GlorotInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Tanh(),
            axes=(action_axes, )
        ),
    ])


def test_dependent_environment():
    environment = gym.make('DependentEnv-v0')

    total_rewards = []
    for i in range(10):
        agent = dqn.Agent(
            dqn.space_shape(environment.observation_space),
            environment.action_space,
            model=model,
            epsilon=dqn.decay_generator(start=1.0, decay=0.995, minimum=0.1),
            gamma=0.99,
            learning_rate=0.1,
        )

        rl_loop.rl_loop_train(environment, agent, episodes=10)

        total_rewards.append(
            rl_loop.evaluate_single_episode(environment, agent)
        )

    # most of these 10 agents will be able to converge to the perfect policy
    assert np.mean(np.array(total_rewards) == 100) >= 0.5


if __name__ == "__main__":
    test_dependent_environment()
