from ngraph.examples.dqn import dqn
import pytest
import numpy as np
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
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    with pytest.raises(ValueError):
        model.predict(np.zeros((1, )))


def test_model_predict_single_shape():
    model = dqn.ModelWrapper(
        state_axes=1, action_size=2, batch_size=3, model=small_model
    )

    # missing axis is assumed to be a single length batch access
    result = model.predict_single(np.zeros((1, )))

    assert result.shape == (2, )


def test_model_predict_single_shape_after_predict():
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

    for _ in range(100):
        model.train(np.zeros((1, 1)), np.ones((2, 1)))

    result_after = model.predict(np.ones((1, 3)))

    assert np.any(np.not_equal(result_before, result_after))


def test_epsilon_linear():
    epsilon_linear = dqn.linear_generator(
        start=0,
        end=100,
        steps=101,
    )

    target = [float(i) for i in range(10)]
    found = [next(epsilon_linear) for _ in range(10)]
    np.testing.assert_allclose(found, target)


def test_epsilon_linear_after_end():
    epsilon_linear = dqn.linear_generator(
        start=0,
        end=2,
        steps=3,
    )

    target = [0, 1, 2, 2, 2]
    found = [next(epsilon_linear) for _ in range(5)]
    np.testing.assert_allclose(found, target)


def test_decay_generator_minimum():
    generator = dqn.decay_generator(1, 1, 2)

    assert next(generator) == 2


def test_decay_generator_simple():
    generator = dqn.decay_generator(1, 0.5, 0.125)

    assert next(generator) == 1
    assert next(generator) == 0.5
    assert next(generator) == 0.25
    assert next(generator) == 0.125
    assert next(generator) == 0.125
