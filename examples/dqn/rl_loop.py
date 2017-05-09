import gym
import simple_environments
from dqn import Agent


def rl_loop(environment, agent, episodes):
    """
    train an agent inside an environment for a set number of episodes
    """
    for episode in range(episodes):
        state = environment.reset()
        done = False
        step = 0
        total_reward = 0
        while not done:
            # environment.render()

            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            agent.observe_results(state, action, reward, next_state, done)

            state = next_state
            step += 1
            total_reward += reward

        agent.end_of_episode()
        print(
            'episode: {}, steps: {}, last_reward: {}, sum(reward): {}'.format(
                episode, step, reward, total_reward
            )
        )


def evaluate_single_episode(environment, agent):
    """
    evaluate a single episode of agent operating inside of an environment
    """
    state = environment.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        # environment.render()

        action = agent.act(state, training=False)
        next_state, reward, done, _ = environment.step(action)

        state = next_state
        step += 1
        total_reward += reward

    agent.end_of_episode()

    return total_reward


def main():
    environment = gym.make('CartPole-v0')
    agent = Agent(environment.observation_space, environment.action_space)

    rl_loop(environment, agent)


if __name__ == "__main__":
    main()
