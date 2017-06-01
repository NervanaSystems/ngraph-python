import ngraph as ng
import pytest
import numpy as np
from ngraph.frontends.neon import dqn
from ngraph.frontends import neon


def small_model(action_axes):
    return neon.Sequential([
        neon.Affine(
            weight_init=neon.GlorotInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Tanh(),
            axes=(action_axes, )
        ),
    ])


def test_model_predict():
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    result = model.predict(np.zeros((1, 3)))

    assert result.shape == (2, 3)


def test_model_predict_wrong_shape():
    # todo: this should now raise an exception
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    # missing axis is assumed to be a single length batch access
    with pytest.raises(ValueError):
        result = model.predict(np.zeros((1, )))


def test_model_predict_single_shape():
    # todo: this should now raise an exception
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    # missing axis is assumed to be a single length batch access
    result = model.predict_single(np.zeros((1, )))

    assert result.shape == (2, )


def test_model_predict_single_shape_after_predict():
    # todo: this should now raise an exception
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    result = model.predict(np.zeros((1, 3)))

    # missing axis is assumed to be a single length batch access
    result = model.predict_single(np.zeros((1, )))

    assert result.shape == (2, )


def test_model_predict_different():
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    result_before = model.predict(np.zeros((1, 3)))
    result_before = np.copy(result_before)
    result_after = model.predict(np.ones((1, 3)))

    assert np.any(np.not_equal(result_before, result_after))


def test_model_train():
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    result_before = model.predict(np.ones((1, 3)))
    result_before = np.copy(result_before)
    print result_before

    for _ in range(100):
        model.train(np.zeros((1, 1)), np.ones((2, 1)))

    result_after = model.predict(np.ones((1, 3)))
    print result_after

    assert np.any(np.not_equal(result_before, result_after))


def make_model(action_axes):
    """
    Given the expected action axes, return a model mapping from observation to
    action axes for use by the dqn agent.
    """
    return neon.Sequential([
        neon.Convolution(
            (8, 8, 32),
            neon.XavierInit(),
            strides=4,
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
        neon.Affine(
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
            batch_norm=True,
            axes=(action_axes, )
        ),
    ])


def test_convolution():
    action_axis = ng.make_axis(name='action', length=5)
    state_axes = ng.make_axes([
        ng.make_axis(4, name='feature'),
        ng.make_axis(84, name='width'),
        ng.make_axis(84, name='height'),
    ])
    batch_axis_1 = ng.make_axis(name='N', length=1)
    batch_axis_all = ng.make_axis(name='N', length=32)

    state_placeholder = ng.placeholder(state_axes + [batch_axis_1])
    state_placeholder_all = ng.placeholder(state_axes + [batch_axis_all])

    model = make_model(action_axis)

    def make_function(placeholder):
        computation = ng.computation(model(placeholder), placeholder)

        transformer = ng.transformers.make_transformer()
        return transformer.add_computation(computation)

    # make a computation using the batch size of 32, though it doesn't need to be used
    # comment out the following line and the batch size 1 works
    make_function(state_placeholder_all)
    print('post')
    # now make another competition with the batch size of 1.
    # the convolution fails because part of ngraph is thinking at the batch size
    # of 32 and part of it is thinking it is a batch size of 1
    make_function(state_placeholder)(np.zeros(state_placeholder.axes.lengths))


def test_epsilon_linear():
    epsilon_linear = dqn.linear_generator(
        start=0,
        end=100,
        steps=101,
    )

    target = [float(i) for i in range(10)]
    found = [epsilon_linear.next() for _ in range(10)]
    np.testing.assert_allclose(found, target)


def test_epsilon_linear_after_end():
    epsilon_linear = dqn.linear_generator(
        start=0,
        end=2,
        steps=3,
    )

    target = [0, 1, 2, 2, 2]
    found = [epsilon_linear.next() for _ in range(5)]
    np.testing.assert_allclose(found, target)


def test_decay_generator_minimum():
    generator = dqn.decay_generator(1, 1, 2)

    assert generator.next() == 2


def test_decay_generator_simple():
    generator = dqn.decay_generator(1, 0.5, 0.125)

    assert generator.next() == 1
    assert generator.next() == 0.5
    assert generator.next() == 0.25
    assert generator.next() == 0.125
    assert generator.next() == 0.125


def test_repeat_memory_append_unable_to_sample():
    memory = dqn.RepeatMemory(3, 10, (1, ))
    memory.append({
        'state': np.array([[1], [2], [3]]),
        'next_state': np.array([[2], [3], [4]]),
        'done': True,
    })

    with pytest.raises(ValueError):
        memory.sample(1)


def test_repeat_memory_append():
    frames_per_observation = 3

    memory = dqn.RepeatMemory(frames_per_observation, 10, (2, ))

    observations = [[i, i] for i in range(10)]
    for i in range(4):
        memory.append({
            'state': np.array(observations[i:i + 3]),
            'next_state': np.array(observations[i + 1:i + 4]),
            'done': False,
        })

    sample = memory.sample(1)

    assert len(sample) == 1
    assert len(sample[0]['state']) == frames_per_observation
    np.testing.assert_allclose(sample[0]['state'], [[3, 3], [4, 4], [5, 5]])
    np.testing.assert_allclose(
        sample[0]['next_state'], [[4, 4], [5, 5], [6, 6]]
    )
    assert sample[0]['done'] == False
