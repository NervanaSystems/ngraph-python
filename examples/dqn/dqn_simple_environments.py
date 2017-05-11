import gym
import simple_environments
from rl_loop import rl_loop
from ngraph.frontends.neon.dqn import Agent


def main():
    environment = gym.make('DependentEnv-v0')
    # environment = gym.make('CartPole-v0')
    # environment = gym.make('SpaceInvaders-v0')
    # todo: specify axes here so that meaningful axes names can be specified such as height and width
    agent = Agent(environment.observation_space, environment.action_space)

    rl_loop(environment, agent, episodes=2000)


if __name__ == "__main__":
    main()
