import numpy as np
import random
from collections import deque

import gym
from gym import spaces

from contextlib import closing
import ngraph as ng
from ngraph.frontends import neon


class Namespace(object):
    pass


def make_axes(lengths, name=None):
    """
    returns an axes of axis objects with length specified by the array `lengths`

    note: this function may be removable if the ngraph version of make_axes is changed
    """
    if isinstance(lengths, ng.Axes):
        return lengths

    if isinstance(lengths, (int, long)):
        lengths = [lengths]

    def make_name(i):
        if name:
            return '{name}_{i}'.format(name=name, i=i)

    return ng.make_axes([
        ng.make_axis(length=length, name=make_name(i))
        for i, length in enumerate(lengths)
    ])


class ModelWrapper(object):
    """the ModelWrapper is responsible for interacting with neon and ngraph"""

    def __init__(self, state_axes, action_size, batch_size, model):
        """
        for now, model must be a function which takes action_axes, and
        returns a neon container
        """
        super(ModelWrapper, self).__init__()

        print(state_axes, action_size, batch_size)
        self.axes = Namespace()
        # todo: standardize axis pattern
        # todo: how to specify which of the axes are which?
        self.axes.state = make_axes(state_axes, name='state')
        self.axes.action = ng.make_axis(name='action', length=action_size)
        self.axes.n = ng.make_axis(name='N', length=batch_size)
        self.axes.n1 = ng.make_axis(name='N', length=1)

        self.state = ng.placeholder(self.axes.state + [self.axes.n])
        self.state_single = ng.placeholder(self.axes.state + [self.axes.n1])
        self.target = ng.placeholder([self.axes.action, self.axes.n])

        # todo: except model as input parameter to constructor
        self.model = model(self.axes.action)

        # construct inference computation
        with neon.Layer.inference_mode_on():
            inference = self.model(self.state)

        self.transformer = ng.transformers.make_transformer()
        self.inference_computation = self.transformer.computation(
            inference, self.state
        )

        # construct inference computation
        with neon.Layer.inference_mode_on():
            inference_single = self.model(self.state_single)

        self.inference_computation_single = self.transformer.computation(
            inference_single, self.state_single
        )

        # construct training computation
        loss = ng.mean(
            ng.squared_L2(self.model(self.state) - self.target), out_axes=()
        )

        optimizer = neon.RMSProp(learning_rate=0.0001)
        train_output = ng.sequential([
            optimizer(loss),
            loss,
        ])

        self.train_computation = self.transformer.computation(
            train_output, self.state, self.target
        )

    def predict_single(self, state):
        """run inference on the model for a single input state"""
        print(state.shape)
        print(self.axes.state.lengths)
        state = state.reshape(self.axes.state.lengths + (1, ))
        # return self.inference_computation_single(state)
        state = np.concatenate([state]*self.axes.n.length, axis=-1)
        return self.inference_computation(state)[..., 0]

    def predict(self, state):
        if state.shape != self.state.axes.lengths:
            raise ValueError((
                'predict received stage with wrong shape. found {}, expected {} '
            ).format(state.shape, self.state.axes.lengths))
        return self.inference_computation(state)

    def train(self, states, targets):
        # todo: check shape
        self.train_computation(states, targets)


def space_shape(space):
    """return the shape of tensor expected for a given space"""
    if isinstance(space, spaces.Discrete):
        return [space.n]
    else:
        return space.shape


class Agent(object):
    """the Agent is responsible for interacting with the environment."""

    def __init__(self, state_axes, action_space, model):
        super(Agent, self).__init__()

        self.update_after_episode = False
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_minimum = 0.1
        self.gamma = 0.9
        self.batch_size = 32
        self.action_space = action_space

        self.memory = Memory(maxlen=100000)
        self.model_wrapper = ModelWrapper(
            state_axes=state_axes,
            action_size=action_space.n,
            batch_size=self.batch_size,
            model=model,
        )

    def act(self, state, training=True):
        """
        given a state, return the index of the action that should be taken

        if training is true, occasionally return a randomly sampled action
        from the action space instead
        """
        if training:
            if self.epsilon > self.epsilon_minimum:
                self.epsilon *= self.epsilon_decay

            if np.random.rand() <= self.epsilon:
                return self.action_space.sample()

        # print(self.model_wrapper.predict(state))
        return np.argmax(self.model_wrapper.predict_single(state))

    def observe_results(self, state, action, reward, next_state, done):
        # print(state, action, reward, next_state)
        if done:
            reward -= 10

        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        if not self.update_after_episode:
            self._update()

    def end_of_episode(self):
        if self.update_after_episode:
            self._update

    def _update(self):
        if len(self.memory) < self.batch_size:
            return

        states = []
        targets = []
        samples = self.memory.sample(self.batch_size)

        print([sample['state'].shape for sample in samples])

        states = np.stack([sample['state'] for sample in samples], axis=-1)
        next_states = np.stack([sample['next_state'] for sample in samples],
                               axis=-1)

        targets = self.model_wrapper.predict(states)
        next_values = self.model_wrapper.predict(next_states)

        for i, sample in enumerate(samples):
            target = sample['reward']

            if not sample['done']:
                target += self.gamma * np.amax(next_values[:, i])

            targets[sample['action'], i] = target

        # print('states', states)
        # print('targets', targets)
        self.model_wrapper.train(states, targets)


class Memory(deque):
    """
    the Memory is used to keep track of what is happened in the past so that
    we can sample from it and learn
    """

    def __init__(self, **kwargs):
        super(Memory, self).__init__(**kwargs)

    def sample(self, batch_size):
        return random.sample(self, batch_size)
