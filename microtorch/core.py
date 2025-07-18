import math

DEBUG = False  # Set to True to enable debug logs


class Value:

    def __init__(self, data, _children=(), _operation='', label=''):
        self.data = data
        self._children = _children
        self._operation = _operation
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

        if DEBUG:
            print(f"[FORWARD] {self}")

    def __repr__(self):
        return f"Value(data={self.data} | grad={self.grad} | op={self._operation} | label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] + : {out} → {self}, {other}")
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] * : {out} → {self}, {other}")
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] ** : {out} → {self}")
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] exp : {out} → {self}")
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return self * -1

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] relu : {out} → {self}")
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] tanh : {out} → {self}")
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            if DEBUG:
                print(f"[BACKWARD] sigmoid : {out} → {self}")
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            if DEBUG:
                print(f"[BACKWARD] Executing backward on: {node}")
            node._backward()

    def zero_grad(self):
        self.grad = 0
        for child in self._children:
            child.zero_grad()


