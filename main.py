from microtorch.neural_net import MLP
from microtorch.core import Value
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate dataset
result = make_regression(n_samples=1000, n_features=3, noise=0, random_state=42, coef=False)
xs = result[0]
ys = result[1]

# Normalize input data
scaler = StandardScaler()
xs = scaler.fit_transform(xs)

# Normalize the targets for better training stability
y_scaler = StandardScaler()
ys_normalized = y_scaler.fit_transform(ys.reshape(-1, 1)).flatten()

# Split data into train/validation (80/20)
split_idx = int(0.8 * len(xs))
train_xs, val_xs = xs[:split_idx], xs[split_idx:]
train_ys, val_ys = ys_normalized[:split_idx], ys_normalized[split_idx:]

# Convert to Value objects
train_xs_val = [[Value(xij) for xij in xi] for xi in train_xs]
train_ys_val = [Value(y) for y in train_ys]
val_xs_val = [[Value(xij) for xij in xi] for xi in val_xs]
val_ys_val = [Value(y) for y in val_ys]

# Create neural network with improved initialization
n = MLP(3, [12, 6, 1])

# Initialize weights with Xavier/Glorot initialization
for layer in n.layers:
    for neuron in layer.neurons:
        fan_in = len(neuron.w)
        std = (2.0 / fan_in) ** 0.5  # Xavier initialization
        for w in neuron.w:
            w.data = random.gauss(0, std)
        neuron.b.data = 0.0

random.seed(42)

# Training hyperparameters
batch_size = 16  # Smaller batch size for better gradients
num_samples = len(train_xs_val)
patience = 50  # Early stopping patience
best_val_loss = float('inf')
patience_counter = 0

# Training loop
train_losses = []
val_losses = []

def evaluate_model(xs_val, ys_val):
    """Evaluate model on given dataset"""
    total_loss = Value(0.0)
    predictions = []
    targets = []
    
    for x, y in zip(xs_val, ys_val):
        pred = n(x)
        pred = pred[0] if isinstance(pred, list) else pred
        total_loss += (pred - y) ** 2
        predictions.append(pred.data)
        targets.append(y.data)
    
    mse = total_loss.data / len(xs_val)
    return mse, predictions, targets

for epoch in range(2000):
    # Shuffle training data
    indices = list(range(num_samples))
    random.shuffle(indices)
    epoch_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Adaptive learning rate with warmup
    if epoch < 10:
        # Warmup phase
        learning_rate = 0.0001 * (epoch + 1) / 10
    else:
        # Exponential decay after warmup
        learning_rate = 0.001 * (0.995 ** ((epoch - 10) // 20))
    
    # Training phase
    n.zero_grad()  # Zero gradients at start of epoch
    
    for batch_idx in range(num_batches):
        batch_indices = indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
        batch_loss = Value(0.0)

        # Accumulate gradients over batch
        for i in batch_indices:
            x = train_xs_val[i]
            y = train_ys_val[i]
            ypred = n(x)
            ypred = ypred[0] if isinstance(ypred, list) else ypred
            loss = (ypred - y) ** 2
            batch_loss += loss

        # Average loss over batch
        batch_loss = batch_loss * (1.0 / len(batch_indices))
        
        # Backward pass (accumulates gradients)
        batch_loss.backward()
        
        epoch_loss += batch_loss.data
    
    # Update parameters once per epoch (after accumulating all gradients)
    # Apply gradient clipping
    total_grad_norm = 0.0
    for p in n.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    # Gradient clipping by norm
    max_grad_norm = 1.0
    if total_grad_norm > max_grad_norm:
        clip_factor = max_grad_norm / total_grad_norm
        for p in n.parameters():
            if p.grad is not None:
                p.grad *= clip_factor
    
    # Parameter update
    for p in n.parameters():
        if p.grad is not None:
            p.data += -learning_rate * p.grad
    
    # Zero gradients for next epoch
    n.zero_grad()
    
    # Calculate average training loss
    avg_train_loss = epoch_loss / num_batches
    
    # Validation evaluation
    val_loss, _, _ = evaluate_model(val_xs_val, val_ys_val)
    
    # Store losses
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    
    # Progress logging
    print(f"Epoch {epoch:4d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {learning_rate:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
        break
    
    # Alternative early stopping condition
    if avg_train_loss < 1e-6:
        print(f"Training converged at epoch {epoch}")
        break

print(f"\nTraining completed after {epoch + 1} epochs")
print(f"Final train loss: {train_losses[-1]:.6f}")
print(f"Final validation loss: {val_losses[-1]:.6f}")

# Final evaluation and accuracy metrics
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

# Evaluate on training set
# Mean Squared Error
# Makes negative and positive errors equal (both bad!)
# Punishes big mistakes more than small ones
# Error of 4 → Penalty of 16 (ouch!)
# Error of 1 → Penalty of 1 (not so bad)

# Good MSE: Close to 0 (perfect predictions)
# Bad MSE: Large numbers (terrible predictions)
train_mse, train_preds, train_targets = evaluate_model(train_xs_val, train_ys_val)
print(f"\nTraining Set:")
print(f"  MSE: {train_mse:.6f}")
print(f"  RMSE: {train_mse**0.5:.6f}")

# Calculate R² score for training set
# 1.0 = Perfect! AI explains 100% of variance 🏆
# 0.9 = Excellent! AI explains 90% of variance 🥇
# 0.5 = Okay, AI explains 50% of variance 📊
# 0.0 = Terrible, AI no better than guessing average 😞
# Negative = AI is worse than just guessing average! 💀
train_targets_np = np.array(train_targets)
train_preds_np = np.array(train_preds)
ss_res_train = np.sum((train_targets_np - train_preds_np) ** 2)  # How wrong our AI is
ss_tot_train = np.sum((train_targets_np - np.mean(train_targets_np)) ** 2)  #  How wrong we'd be if we just guessed the average every time
r2_train = 1 - (ss_res_train / ss_tot_train)
print(f"  R² Score: {r2_train:.6f}")

# Mean Absolute Error
# Errors: [1, 1, 1, 10]
# MAE = (1 + 1 + 1 + 10) / 4 = 3.25
# MSE = (1 + 1 + 1 + 100) / 4 = 25.75
# MAE is more "forgiving" of big outliers!
mae_train = np.mean(np.abs(train_targets_np - train_preds_np))
print(f"  MAE: {mae_train:.6f}")

# Evaluate on validation set
val_mse, val_preds, val_targets = evaluate_model(val_xs_val, val_ys_val)
print(f"\nValidation Set:")
print(f"  MSE: {val_mse:.6f}")
print(f"  RMSE: {val_mse**0.5:.6f}")

# Calculate R² score for validation set
val_targets_np = np.array(val_targets)
val_preds_np = np.array(val_preds)
ss_res_val = np.sum((val_targets_np - val_preds_np) ** 2)
ss_tot_val = np.sum((val_targets_np - np.mean(val_targets_np)) ** 2)
r2_val = 1 - (ss_res_val / ss_tot_val)
print(f"  R² Score: {r2_val:.6f}")

# Mean Absolute Error
mae_val = np.mean(np.abs(val_targets_np - val_preds_np))
print(f"  MAE: {mae_val:.6f}")

# Accuracy within tolerance (for regression)
# How often your model's predictions are "close enough" to the actual values - a custom accuracy metric for regression problems.
tolerance = 0.1  # 10% of standard deviation
std_val = np.std(val_targets_np)
tolerance_abs = tolerance * std_val
accuracy_within_tolerance = np.mean(np.abs(val_targets_np - val_preds_np) <= tolerance_abs) * 100
print(f"  Accuracy within {tolerance*100}% tolerance: {accuracy_within_tolerance:.1f}%")

# Print sample predictions
print(f"\nSample Predictions (Validation Set - first 10):")
print("-" * 50)
for i in range(min(10, len(val_targets))):
    target_orig = y_scaler.inverse_transform([[val_targets[i]]])[0][0]
    pred_orig = y_scaler.inverse_transform([[val_preds[i]]])[0][0]
    error = abs(target_orig - pred_orig)
    print(f"Sample {i+1:2d}: Target = {target_orig:8.2f}, Predicted = {pred_orig:8.2f}, Error = {error:6.2f}")

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training curves
ax1.plot(train_losses, label='Training Loss', alpha=0.8)
ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predictions vs Targets (Validation)
ax2.scatter(val_targets_np, val_preds_np, alpha=0.6, s=30)
min_val, max_val = min(min(val_targets_np), min(val_preds_np)), max(max(val_targets_np), max(val_preds_np))
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
ax2.set_xlabel('True Values')
ax2.set_ylabel('Predictions')
ax2.set_title(f'Predictions vs True Values (R² = {r2_val:.4f})')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
residuals = val_targets_np - val_preds_np
ax3.scatter(val_preds_np, residuals, alpha=0.6, s=30)
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
ax3.set_xlabel('Predictions')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model complexity analysis
total_params = len(n.parameters())
print(f"\nModel Complexity:")
print(f"  Total parameters: {total_params}")
print(f"  Architecture: {[len(layer.neurons) for layer in n.layers]}")

# Final gradient analysis
print(f"\nFinal Gradient Analysis:")
grad_magnitudes = [abs(p.grad) if p.grad is not None else 0 for p in n.parameters()]
if grad_magnitudes:
    print(f"  Max gradient magnitude: {max(grad_magnitudes):.6f}")
    print(f"  Min gradient magnitude: {min(grad_magnitudes):.6f}")
    print(f"  Mean gradient magnitude: {np.mean(grad_magnitudes):.6f}")
    print(f"  Std gradient magnitude: {np.std(grad_magnitudes):.6f}")