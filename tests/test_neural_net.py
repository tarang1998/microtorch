from microtorch.neural_net import Neuron, Layer, MLP
from microtorch.core import Value


def test_neuron_forward_and_parameters():
    n = Neuron(3)
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    out = n(x)
    assert isinstance(out, Value)
    params = n.parameters()
    assert len(params) == 4  # 3 weights + 1 bias


def test_neuron_linear():
    n = Neuron(2, nonlin=False)
    x = [Value(1.0), Value(2.0)]
    out = n(x)
    assert isinstance(out, Value)


def test_layer_forward_and_parameters():
    l = Layer(2, 3)
    x = [Value(1.0), Value(2.0)]
    out = l(x)
    assert isinstance(out, list)
    assert len(out) == 3
    params = l.parameters()
    assert len(params) == 3 * 3  # 3 neurons, each with 2 weights + 1 bias


def test_layer_single_output():
    l = Layer(2, 1)
    x = [Value(1.0), Value(2.0)]
    out = l(x)
    assert isinstance(out, Value)


def test_mlp_forward_and_parameters():
    mlp = MLP(2, [3, 2, 1])
    x = [Value(1.0), Value(2.0)]
    out = mlp(x)
    assert isinstance(out, Value)
    params = mlp.parameters()
    # 2->3: 3*(2+1), 3->2: 2*(3+1), 2->1: 1*(2+1)
    expected_param_count = 3*3 + 2*4 + 1*3
    assert len(params) == expected_param_count


def test_repr_methods():
    n = Neuron(2)
    l = Layer(2, 2)
    mlp = MLP(2, [2, 1])
    assert 'Neuron' in repr(n)
    assert 'Layer' in repr(l)
    assert 'MLP' in repr(mlp) 