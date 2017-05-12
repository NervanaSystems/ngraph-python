import gym
import simple_environments
from ngraph.frontends.neon import dqn, rl_loop
from ngraph.frontends import neon


def small_model(action_axes):
    return neon.Sequential([
        neon.Affine(
            nout=20,
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
    agent = dqn.Agent(
        dqn.space_shape(environment.observation_space),
        environment.action_space,
        model=small_model
    )

    rl_loop.rl_loop(environment, agent, episodes=20)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)

    assert total_reward >= 95


if __name__ == "__main__":
    test_dependent_environment()
