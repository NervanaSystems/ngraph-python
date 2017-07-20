import gym
import ngraph as ng
import dqn
import rl_loop
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from examples.dqn.gym_wrapper import ReshapeWrapper, ClipRewardWrapper
from examples.dqn.gym_wrapper import RepeatWrapper, TerminateOnEndOfLifeWrapper
from examples.dqn.gym_wrapper import DimShuffleWrapper
from ngraph.frontends import neon

# factory = ng.transformers.make_transformer_factory('gpu')
# mg.transformers.set_transformer_factory(factory)


def model(action_axes):
    """
    Given the expected action axes, return a model mapping from observation to
    action axes for use by the dqn agent.
    """
    return neon.Sequential([
        neon.Preprocess(lambda x: x / 255.0),
        neon.Convolution(
            (8, 8, 32),
            neon.XavierInit(),
            strides=4,
            activation=neon.Rectlin(),
        ),
        neon.Convolution(
            (4, 4, 64),
            neon.XavierInit(),
            strides=2,
            activation=neon.Rectlin(),
        ),
        neon.Convolution(
            (3, 3, 64),
            neon.XavierInit(),
            strides=1,
            activation=neon.Rectlin(),
        ),
        neon.Affine(
            nout=512,
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
        ),
        neon.Affine(
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=None,
            axes=(action_axes, )
        ),
    ])


def main():
    if False:
        # deterministic version 4 results in a frame skip of 4 and no repeat action probability
        environment = gym.make('BreakoutDeterministic-v4')
        environment = TerminateOnEndOfLifeWrapper(environment)
        environment = ReshapeWrapper(environment)
        environment = ClipRewardWrapper(environment)
        environment = RepeatWrapper(environment, frames=4)
    else:
        # use the environment wrappers found in openai baselines.
        environment = gym.make('BreakoutNoFrameskip-v4')
        environment = wrap_dqn(environment)
        environment = DimShuffleWrapper(environment)

    # todo: perhaps these should be defined in the environment itself
    state_axes = ng.make_axes([
        ng.make_axis(environment.observation_space.shape[0], name='C'),
        ng.make_axis(environment.observation_space.shape[1], name='H'),
        ng.make_axis(environment.observation_space.shape[2], name='W'),
    ])

    agent = dqn.Agent(
        state_axes,
        environment.action_space,
        model=model,
        epsilon=dqn.linear_generator(start=1.0, end=0.1, steps=1000000),
        gamma=0.99,
        learning_rate=0.00025,
        memory=dqn.Memory(maxlen=1000000),
        target_network_update_frequency=1000,
        learning_starts=10000,
    )

    rl_loop.rl_loop_train(environment, agent, episodes=200000)


if __name__ == "__main__":
    main()
