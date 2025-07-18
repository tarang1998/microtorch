from microtorch.neural_net import MLP
from microtorch.core import Value

# Training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Convert inputs to Value objects
xs_val = [[Value(xij) for xij in xi] for xi in xs]
ys_val = [Value(y) for y in ys]

# Create a neural network: 3 inputs, 1 hidden layer with 4 neurons, 1 output
n = MLP(3, [4,5,2,1])

# Training loop
for k in range(10000):
    # Forward pass
    ypred = [n(x) for x in xs_val]
    # Ensure ypred is a list of Value, not list of list
    ypred = [y[0] if isinstance(y, list) else y for y in ypred]
    loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys_val, ypred)), start=Value(0.0))

    # Zero gradients
    for p in n.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Update parameters
    for p in n.parameters():
        p.data += -0.005 * p.grad

    if k % 10 == 0 or k == 999:
        print(f"Step {k}: Loss = {loss.data}")

# Print final predictions
print("Final predictions:")
for x, y in zip(xs_val, ys):
    pred = n(x)
    if isinstance(pred, list):
        pred = pred[0]
    print(f"Input: {[v.data for v in x]}, Target: {y}, Predicted: {pred.data}")
