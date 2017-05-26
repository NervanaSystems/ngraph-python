import gym
from ngraph.frontends.neon.dqn import Agent


def rl_loop(environment, agent, episodes, render=False):
    """
    train an agent inside an environment for a set number of episodes

    # todo: rename to rl_loop_train, and follow the same pattern and callbacks
    # neon.loop_train used for supervised learning.
    """
    total_steps = 0
    for episode in range(episodes):
        state = environment.reset()
        done = False
        step = 0
        total_reward = 0
        while not done:
            if render:
                environment.render()

            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            agent.observe_results(state, action, reward, next_state, done)

            state = next_state
            step += 1
            total_steps += 1
            total_reward += reward

            if total_steps % 50000:
                print(
                    'evaluation episode total reward: {}'.format(
                        evaluate_single_episode(environment, agent, render)
                    )
                )

        agent.end_of_episode()
        print(
            'episode: {}, total_steps: {}, steps: {}, last_reward: {}, '
            'sum(reward): {}'.format(
                episode, total_steps, step, reward, total_reward
            )
        )


def evaluate_single_episode(environment, agent, render=False):
    """
    evaluate a single episode of agent operating inside of an environment
    """
    state = environment.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        if render:
            environment.render()

        action = agent.act(state, training=False)
        next_state, reward, done, _ = environment.step(action)

        state = next_state
        step += 1
        total_reward += reward

    agent.end_of_episode()

    return total_reward
