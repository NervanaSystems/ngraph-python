import gym
import simple_environments
from rl_loop import rl_loop
from ngraph.frontends.neon import dqn
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


def main():
    # environment = gym.make('DependentEnv-v0')
    environment = gym.make('CartPole-v0')
    # environment = gym.make('SpaceInvaders-v0')
    # todo: specify axes here so that meaningful axes names can be specified such as height and width
    agent = dqn.Agent(
        dqn.space_shape(environment.observation_space),
        environment.action_space,
        model=model,
        epsilon=dqn.decay_generator(start=1.0, decay=0.995, minimum=0.1),
        gamma=0.9,
    )

    rl_loop(environment, agent, episodes=50)


if __name__ == "__main__":
    main()
