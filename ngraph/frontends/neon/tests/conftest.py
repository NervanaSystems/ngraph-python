import pytest
import ngraph as ng


@pytest.fixture
def batch_axis(batch_size):
    return ng.make_axis(length=batch_size, name='N')


@pytest.fixture
def recurrent_axis(sequence_length):
    return ng.make_axis(length=sequence_length, name='REC')


@pytest.fixture
def input_axis(input_size):
    return ng.make_axis(length=input_size, name='input')


@pytest.fixture
def output_axis(output_size):
    return ng.make_axis(length=output_size, name="hidden")


@pytest.fixture
def input_placeholder(input_axis, batch_axis):
    return ng.placeholder([input_axis, batch_axis])


@pytest.fixture
def recurrent_input(input_axis, recurrent_axis, batch_axis):
    return ng.placeholder([input_axis, recurrent_axis, batch_axis])
