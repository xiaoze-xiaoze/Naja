import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("Python (scikit-learn) Linear Regression Benchmark")
    
    # 1. Setup problem size
    n_samples = 100_000
    n_features = 50
    noise_std = 1.0
    
    print(f"Dataset size: {n_samples} samples, {n_features} features")

    # 2. Generate synthetic data
    np.random.seed(42)
    
    # Generate random true weights
    true_weights = np.random.uniform(-5.0, 5.0, n_features)
    true_intercept = np.random.uniform(-10.0, 10.0)

    # Generate features X
    X = np.random.uniform(-10.0, 10.0, (n_samples, n_features))
    
    # Generate target y
    y = np.dot(X, true_weights) + true_intercept
    # Add noise
    y += np.random.normal(0, noise_std, n_samples)

    # Split into train/test (80/20)
    n_train = int(n_samples * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Data generation complete. Training on {n_train} samples...")

    # 3. Train model
    start_time = time.time()
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    end_time = time.time()
    
    duration_ms = (end_time - start_time) * 1000
    print(f"Training completed in {duration_ms:.2f}ms")

    # 4. Evaluate
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance (Test Set):")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² : {r2:.6f}")

if __name__ == "__main__":
    main()
