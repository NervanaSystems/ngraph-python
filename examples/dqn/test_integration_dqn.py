import gym
import simple_environments
from ngraph.frontends.neon import dqn, rl_loop
from ngraph.frontends import neon


def model(action_axes):
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
        model=model,
        epsilon=dqn.decay_generator(start=1.0, decay=0.995, minimum=0.1)
    )

    rl_loop.rl_loop(environment, agent, episodes=20)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)

    assert total_reward >= 95


if __name__ == "__main__":
    test_dependent_environment()
