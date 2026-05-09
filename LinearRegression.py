import numpy as np

# -----------------------------
# DATASET
# -----------------------------
x1 = np.array([60, 80, 100, 120, 90, 70], dtype=float)
x2 = np.array([5, 10, 3, 15, 8, 12], dtype=float)
y  = np.array([45, 55, 75, 80, 65, 50], dtype=float)

# Normalize (add before training)
x1 = (x1 - np.mean(x1)) / np.std(x1)
x2 = (x2 - np.mean(x2)) / np.std(x2)

N = len(y)

# -----------------------------
# LOSS FUNCTION
# -----------------------------
def compute_loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# -----------------------------
# GRADIENT DESCENT (NO MOMENTUM)
# -----------------------------
def gradient_descent(alpha=0.001, iterations=100):
    w1, w2, b = 0.0, 0.0, 0.0
    
    print("\n--- Gradient Descent (No Momentum) ---")
    
    for it in range(iterations):
        
        # Predictions
        y_hat = w1 * x1 + w2 * x2 + b
        
        # Error
        error = y - y_hat
        
        # Gradients
        dw1 = -(2/N) * np.sum(error * x1)
        dw2 = -(2/N) * np.sum(error * x2)
        db  = -(2/N) * np.sum(error)
        
        # 🔧 Gradient clipping (prevents explosion)
        max_grad = 1e4
        dw1 = np.clip(dw1, -max_grad, max_grad)
        dw2 = np.clip(dw2, -max_grad, max_grad)
        db  = np.clip(db,  -max_grad, max_grad)
        
        # Update
        w1 = w1 - alpha * dw1
        w2 = w2 - alpha * dw2
        b  = b  - alpha * db
        
        # Divergence check
        if np.isnan(w1) or np.isinf(w1):
            print("Diverged!")
            break
        
        # Track loss every 10 iterations
        if (it + 1) % 10 == 0:
            loss = compute_loss(y, y_hat)
            print(f"Iter {it+1}: Loss = {loss:.6f}")
    
    print("\nFinal Weights (No Momentum):")
    print(f"w1 = {w1:.6f}")
    print(f"w2 = {w2:.6f}")
    print(f"b  = {b:.6f}")
    
    return w1, w2, b

# -----------------------------
# GRADIENT DESCENT WITH MOMENTUM
# -----------------------------
def gradient_descent_momentum(alpha=0.001, beta=0.09, iterations=100):
    w1, w2, b = 0.0, 0.0, 0.0
    v_w1, v_w2, v_b = 0.0, 0.0, 0.0
    
    print(f"\n--- Gradient Descent with Momentum (beta={beta}) ---")
    
    for it in range(iterations):
        
        # Predictions
        y_hat = w1 * x1 + w2 * x2 + b
        
        # Error
        error = y - y_hat
        
        # Gradients
        dw1 = -(2/N) * np.sum(error * x1)
        dw2 = -(2/N) * np.sum(error * x2)
        db  = -(2/N) * np.sum(error)
        
        # 🔧 Gradient clipping
        max_grad = 1e4
        #dw1 = np.clip(dw1, -max_grad, max_grad)
        #dw2 = np.clip(dw2, -max_grad, max_grad)
        #db  = np.clip(db,  -max_grad, max_grad)
        
        # Velocity update
        v_w1 = beta * v_w1 - alpha * dw1
        v_w2 = beta * v_w2 - alpha * dw2
        v_b  = beta * v_b  - alpha * db
        
        # Weight update
        w1 = w1 + v_w1
        w2 = w2 + v_w2
        b  = b  + v_b
        
        # Divergence check
        if np.isnan(w1) or np.isinf(w1):
            print("Diverged!")
            break
        
        # Track loss every 10 iterations
        if (it + 1) % 10 == 0:
            loss = compute_loss(y, y_hat)
            print(f"Iter {it+1}: Loss = {loss:.6f}")
    
    print("\nFinal Weights (Momentum β=0.09):")
    print(f"w1 = {w1:.6f}")
    print(f"w2 = {w2:.6f}")
    print(f"b  = {b:.6f}")
    
    return w1, w2, b

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    
    # Run standard GD
    gradient_descent()
    
    # Run GD with momentum (β = 0.09)
    gradient_descent_momentum()