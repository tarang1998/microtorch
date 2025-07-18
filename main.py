from microtorch.neural_net import MLP
from microtorch.core import Value
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

result = make_regression(n_samples=1000, n_features=3, noise=0, random_state=42, coef=False)
xs = result[0]
ys = result[1]


# Convert inputs to Value objects
xs_val = [[Value(xij) for xij in xi] for xi in xs]
ys_val = [Value(y) for y in ys]

# Create a neural network: 3 inputs, 2 hidden layers with 16 neurons each, 1 output
n = MLP(3, [4,16,32,3,1])

# Set random seed for reproducibility
random.seed(42)

batch_size = 100
num_samples = len(xs_val)

# Training loop
losses = []
for k in range(2000):
    # Shuffle data for each epoch
    indices = list(range(num_samples))
    random.shuffle(indices)
    epoch_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        batch_indices = indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
        loss = Value(0.0)
        for i in batch_indices:
            x = xs_val[i]
            y = ys_val[i]
            ypred = n(x)
            ypred = ypred[0] if isinstance(ypred, list) else ypred
            loss += ((ypred - y) ** 2)
        # Zero gradients
        n.zero_grad()
        # Backward pass
        loss.backward()
        # Update parameters
        learning_rate = 0.001
        for p in n.parameters():
            p.data += -learning_rate * p.grad
        epoch_loss += loss.data
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Step {k}: Avg Loss = {avg_epoch_loss}")
    if k % 10 == 0 or k == 999:
        losses.append(avg_epoch_loss)
    # Early stopping
    if avg_epoch_loss < 1e-20:
        print(f"Early stopping at step {k}")
        break

# Print final predictions
print("Final predictions:")
for x, y in zip(xs_val, ys):
    pred = n(x)
    if isinstance(pred, list):
        pred = pred[0]
    print(f"Input: {[v.data for v in x]}, Target: {y}, Predicted: {pred.data}")

# Plot loss
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Avg Loss')
plt.show()
