import gym
import simple_environments
import rl_loop
from dqn import Agent


def test_dependent_environment():
    environment = gym.make('DependentEnv-v0')
    agent = Agent(environment.observation_space, environment.action_space)

    rl_loop.rl_loop(environment, agent, episodes=20)

    total_reward = rl_loop.evaluate_single_episode(environment, agent)

    assert total_reward >= 95


if __name__ == "__main__":
    main()
