import gym
import ngraph as ng
from ngraph.frontends.neon import dqn, rl_loop
from ngraph.frontends import neon


def model(action_axes):
    return neon.Sequential([
        neon.Affine(
            nout=20,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Tanh(),
            batch_norm=True,
        ),
        neon.Affine(
            nout=20,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Tanh(),
            batch_norm=True,
        ),
        neon.Affine(
            nout=20,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Tanh(),
            batch_norm=True,
        ),
        neon.Affine(
            nout=2,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
    ])


def main():
    # initialize gym environment 
    #environment = gym.make('Pong-v0')
    #print environment.observation_space
    environment = gym.make('CartPole-v0')
    print environment.observation_space

    state_axes = ng.make_axes([
        ng.make_axis(environment.observation_space.shape[0], name='width')
    ])

    agent = dqn.Agent(state_axes, environment.action_space, model=model)

    rl_loop.rl_loop(environment, agent, episodes=1000)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)
    print total_reward


if __name__ == "__main__":
    main()
