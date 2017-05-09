import ngraph as ng
import pytest
import numpy as np
import dqn
from ngraph.frontends import neon


def small_model(action_axes):
    return neon.Sequential([
        # neon.Affine(
        #     nout=20,
        #     weight_init=neon.GlorotInit(),
        #     bias_init=neon.ConstantInit(),
        #     activation=neon.Tanh(),
        # ),
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
