import math
from microtorch.core import Value


def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0

def test_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0

def test_power():
    a = Value(2.0)
    b = a ** 3
    assert b.data == 8.0
    b.backward()
    assert math.isclose(a.grad, 12.0)

def test_neg_and_sub():
    a = Value(5.0)
    b = Value(2.0)
    c = a - b
    assert c.data == 3.0
    d = -a
    assert d.data == -5.0

def test_division():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    assert c.data == 3.0
    c.backward()
    assert math.isclose(a.grad, 0.5)
    assert math.isclose(b.grad, -1.5)

def test_exp():
    a = Value(1.0)
    b = a.exp()
    assert math.isclose(b.data, math.exp(1.0))
    b.backward()
    assert math.isclose(a.grad, math.exp(1.0))

def test_relu():
    a = Value(-1.0)
    b = a.relu()
    assert b.data == 0
    a2 = Value(2.0)
    b2 = a2.relu()
    assert b2.data == 2.0
    b2.backward()
    assert a2.grad == 1.0

def test_tanh():
    a = Value(0.5)
    b = a.tanh()
    expected = (math.exp(1.0) - 1) / (math.exp(1.0) + 1)
    assert math.isclose(b.data, expected)
    b.backward()
    assert math.isclose(a.grad, 1 - b.data ** 2)

def test_sigmoid():
    a = Value(0.0)
    b = a.sigmoid()
    assert math.isclose(b.data, 0.5)
    b.backward()
    assert math.isclose(a.grad, 0.25)

def test_chain_rule():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a + 2
    c.backward()
    assert math.isclose(a.grad, 4.0)
    assert math.isclose(b.grad, 2.0)

def test_zero_grad():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    a.zero_grad()
    assert a.grad == 0
