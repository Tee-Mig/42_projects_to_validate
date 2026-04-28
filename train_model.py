import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import tempfile

BASE_DIR = "datasets"

def run_error_tests() -> None:
    tests = []

    # (name, file, epochs, lr, should_fail)

    tests.append(("Dataset not found", "does_not_exist.csv", 1000, 0.01, True))

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. empty csv
        empty_csv = os.path.join(tmpdir, "empty.csv")
        pd.DataFrame().to_csv(empty_csv, index=False)
        tests.append(("empty dataset", empty_csv, 1000, 0.01, True))

        # 2. missing columns
        missing_cols_csv = os.path.join(tmpdir, "missing_cols.csv")
        pd.DataFrame({"km": [10000], "value": [5000]}).to_csv(missing_cols_csv, index=False)
        tests.append(("missing columns", missing_cols_csv, 1000, 0.01, True))

        # 3. non numeric km
        bad_km_csv = os.path.join(tmpdir, "bad_km.csv")
        pd.DataFrame({"km": ["abc", 20000], "price": [5000, 6000]}).to_csv(bad_km_csv, index=False)
        tests.append(("non numeric km", bad_km_csv, 1000, 0.01, True))

        # 4. non numeric price
        bad_price_csv = os.path.join(tmpdir, "bad_price.csv")
        pd.DataFrame({"km": [10000, 20000], "price": [5000, "mig"]}).to_csv(bad_price_csv, index=False)
        tests.append(("non numeric price", bad_price_csv, 1000, 0.01, True))

        # 5. std km = 0
        same_km_csv = os.path.join(tmpdir, "same_km.csv")
        pd.DataFrame({"km": [10000, 10000, 10000], "price": [4000, 5000, 6000]}).to_csv(same_km_csv, index=False)
        tests.append(("km std = 0", same_km_csv, 1000, 0.01, True))

        # 6. std price = 0
        same_price_csv = os.path.join(tmpdir, "same_price.csv")
        pd.DataFrame({"km": [10000, 20000, 30000], "price": [5000, 5000, 5000]}).to_csv(same_price_csv, index=False)
        tests.append(("price std = 0", same_price_csv, 1000, 0.01, True))

        # 7. epochs <= 0
        valid_csv = os.path.join(tmpdir, "data.csv")
        pd.DataFrame({"km": [10000, 20000, 30000], "price": [7000, 6000, 5000]}).to_csv(valid_csv, index=False)
        tests.append(("epochs <= 0", valid_csv, 0, 0.01, True))

        # 8. alpha(learning rate) <= 0
        tests.append(("alpha(learning rate) <= 0", valid_csv, 1000, 0.0, True))

        # 9. km or price < 0
        valid_csv = os.path.join(tmpdir, "negative_value.csv")
        pd.DataFrame({"km": [10000, -9000], "price": [-7000, 6000]}).to_csv(valid_csv, index=False)
        tests.append(("km or price < 0", valid_csv, 1000, 0.01, True))

        # --- RUN TESTS ---
        for name, file, epochs, lr, should_fail in tests:
            print(f"\n--- Test: {name} ---")

            try:
                train_model(file, epochs, lr)

                if should_fail:
                    print("❌ expected error, but none occurred")
                else:
                    print("✅")

            except Exception as e:
                if should_fail:
                    print(f"✅ caught expected error: {e}")
                else:
                    print(f"❌ unexpected error: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear regression model")

    parser.add_argument("--file", default="data.csv", help="Dataset file (default: data.csv)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")

    return parser.parse_args()

def train_model(file: str, epochs: int, alpha: float):
    file = os.path.join(BASE_DIR, file)
    if not os.path.exists(file):
        raise FileNotFoundError("Dataset not found")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

    if df.empty:
        raise ValueError("Dataset is empty")

    required_columns = {"km", "price"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain 'km' and 'price' columns")
    
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    
    if alpha <= 0:
        raise ValueError("learning rate must be positive")

    df["km"] = pd.to_numeric(df["km"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if df["km"].isna().any() or df["price"].isna().any():
        raise ValueError("Columns 'km' and 'price' must contain only numeric values")
    
    if (df["km"] < 0).any():
        raise ValueError("Column 'km' must contain only non-negative values")

    if (df["price"] < 0).any():
        raise ValueError("Column 'price' must contain only non-negative values")

    x = df["km"].values
    y = df["price"].values

    x_mean = df["km"].mean()
    x_std = df["km"].std()

    y_mean = df["price"].mean()
    y_std = df["price"].std()

    if x_std == 0 or y_std == 0:
        raise ValueError("Std is 0, impossible to normalize")

    # normalization
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # weights to save
    theta0_norm = 0
    theta1_norm = 0

    m = len(x)
    prev_cost = None

    # keep track of gradients
    grad0_history = []
    grad1_history = []
    cost_history = []

    for epoch in range(epochs):
        estimated_price_norm = theta0_norm + (theta1_norm * x_norm)
        error = estimated_price_norm - y_norm

        # Gradients
        grad0 = (1 / m) * error.sum()
        grad1 = (1 / m) * (error * x_norm).sum()

        # Update thetas
        tmp_theta0 = theta0_norm - alpha * grad0
        tmp_theta1 = theta1_norm - alpha * grad1

        # Compute cost
        current_cost = ((error ** 2).sum()) / (2 * m)

        grad0_history.append(grad0)
        grad1_history.append(grad1)
        cost_history.append(current_cost)

        # Early stopping
        if prev_cost is not None:
            if prev_cost != 0 and abs(prev_cost - current_cost) / prev_cost < 1e-6:
                print(f"Stopped early at epoch {epoch}, cost={current_cost:.3f}")
                break

        prev_cost = current_cost

        theta0_norm = tmp_theta0
        theta1_norm = tmp_theta1

    # denormalization
    theta0 = y_mean + (y_std * theta0_norm) - ((y_std * theta1_norm * x_mean) / x_std)
    theta1 = (y_std / x_std) * theta1_norm

    print("Trained model:")
    print(f"theta0 = {theta0:.2f}")
    print(f"theta1 = {theta1:.2f}")
    print(f"cost = {prev_cost}")

    with open("thetas.txt", "w") as f:
        f.write(f"{theta0},{theta1}")

    _, axes = plt.subplots(3, 1, figsize=(8, 10))

    y_pred = theta0 + (theta1 * x)

    # plot 1: data + model
    axes[0].scatter(x, df["price"], color="blue", label="Data")

    idx = x.argsort()
    axes[0].plot(x[idx], y_pred[idx], color="red", label="Model")

    axes[0].set_xlabel("Mileage")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].set_title("Linear regression")

    # plot 2: gradients
    axes[1].plot(grad0_history, label="grad0")
    axes[1].plot(grad1_history, label="grad1")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gradient value")
    axes[1].legend()
    axes[1].set_title("Gradient evolution")

    # plot 3: cost
    axes[2].plot(cost_history)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Cost")
    axes[2].set_title("Cost evolution")

    plt.tight_layout()
    plt.show()

def main():
    # print("------ ERROR TESTS ------")
    # run_error_tests()

    try:
        # print("\n\n------ SUBJECT ------\n")
        args = parse_args()
        train_model(args.file, args.epochs, args.lr)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    return 0
    
if __name__ == "__main__":
    main()