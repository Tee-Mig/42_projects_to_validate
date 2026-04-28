import numpy as np
import pandas as pd
import argparse
import tempfile
import os

BASE_DIR = "datasets"

def run_error_tests() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:

        # theta files
        missing_theta = os.path.join(tmpdir, "missing_thetas.txt")

        empty_theta = os.path.join(tmpdir, "empty_thetas.txt")
        with open(empty_theta, "w", encoding="utf-8") as f:
            f.write("")

        bad_format_theta = os.path.join(tmpdir, "bad_format_thetas.txt")
        with open(bad_format_theta, "w", encoding="utf-8") as f:
            f.write("mig")

        bad_numeric_theta = os.path.join(tmpdir, "bad_numeric_thetas.txt")
        with open(bad_numeric_theta, "w", encoding="utf-8") as f:
            f.write("mig,mig")

        valid_theta = os.path.join(tmpdir, "valid_thetas.txt")
        with open(valid_theta, "w", encoding="utf-8") as f:
            f.write("8490.37,-0.02")

        # csv files
        missing_csv = os.path.join(tmpdir, "missing.csv")

        empty_csv = os.path.join(tmpdir, "empty.csv")
        pd.DataFrame().to_csv(empty_csv, index=False)

        missing_cols_csv = os.path.join(tmpdir, "missing_cols.csv")
        pd.DataFrame({"distance": [10000]}).to_csv(missing_cols_csv, index=False)

        bad_km_csv = os.path.join(tmpdir, "bad_km.csv")
        pd.DataFrame({"km": ["mig", 20000], "price": [5000, 6000]}).to_csv(bad_km_csv, index=False)

        bad_price_csv = os.path.join(tmpdir, "bad_price.csv")
        pd.DataFrame({"km": [10000, 20000], "price": [5000, "mig"]}).to_csv(bad_price_csv, index=False)

        negative_value_csv = os.path.join(tmpdir, "negative_value.csv")
        pd.DataFrame({"km": [10000, -20000], "price": [-5000, 4000]}).to_csv(negative_value_csv, index=False)

        valid_csv = os.path.join(tmpdir, "valid.csv")
        pd.DataFrame({"km": [10000, 20000, 30000], "price": [7000, 6000, 5000]}).to_csv(valid_csv, index=False)

        # --- TESTS ---
        tests = [
            ("file not found", missing_csv, valid_theta, True),
            ("empty dataset", empty_csv, valid_theta, True),
            ("missing columns", missing_cols_csv, valid_theta, True),
            ("non numeric km", bad_km_csv, valid_theta, True),
            ("non numeric price", bad_price_csv, valid_theta, True),
            ("km or price with negatives values", negative_value_csv, valid_theta, True),
            ("missing thetas file", valid_csv, missing_theta, True),
            ("empty thetas file", valid_csv, empty_theta, True),
            ("bad thetas format", valid_csv, bad_format_theta, True),
            ("non numeric thetas", valid_csv, bad_numeric_theta, True),
            ("valid dataset", valid_csv, valid_theta, False),
        ]

        for name, csv_file, theta_file, should_fail in tests:
            print(f"\n--- Test: {name} ---")

            try:
                result = evaluate(csv_file, theta_file)

                if should_fail:
                    if result != 0:
                        print("✅ expected error")
                    else:
                        print("❌ expected failure but got success")
                else:
                    if result == 0:
                        print("✅ success")
                    else:
                        print("❌ unexpected failure")

            except Exception as e:
                print(f"❌ unexpected exception: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate linear regression model")

    parser.add_argument(
        "--file",
        type=str,
        default="data.csv",
        help="Dataset file (default: data.csv)",
    )

    parser.add_argument(
        "--thetas",
        type=str,
        default="thetas.txt",
        help="Theta parameters file (default: thetas.txt)",
    )

    return parser.parse_args()


def evaluate(file: str, theta_file: str) -> int:
    file = os.path.join(BASE_DIR, file)

    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return 1

    if df.empty:
        print("Error: dataset is empty")
        return 1

    if "km" not in df.columns or "price" not in df.columns:
        print("Error: dataset must contain 'km' and 'price' columns")
        return 1


    df["km"] = pd.to_numeric(df["km"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if df["km"].isna().any() or df["price"].isna().any():
        print("Columns 'km' and 'price' must contain only numeric values")
        return 1
    
    if (df["km"] < 0).any():
        print("Column 'km' must contain only non-negative values")
        return 1

    if (df["price"] < 0).any():
        print("Column 'price' must contain only non-negative values")
        return 1

    x = df["km"].to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)

    try:
        with open(theta_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise ValueError("Thetas file is empty")

        parts = [p.strip() for p in content.split(",")]
        if len(parts) != 2:
            raise ValueError("Invalid thetas format. Expected: theta0,theta1")

        theta0 = float(parts[0])
        theta1 = float(parts[1])
    except Exception as e:
        print(f"Error reading thetas: {e}")
        return 1

    # predictions
    y_pred = theta0 + theta1 * x

    # metrics
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    err_tot = float(np.sum((y - y_pred) ** 2))
    var_tot = float(np.sum((y - np.mean(y)) ** 2))

    if var_tot == 0:
        r2 = 1 if err_tot == 0 else 0
    else:
        r2 = 1 - (err_tot / var_tot)

    print("Model metrics:")
    print(f"MAE  = {mae:.2f} €")
    print(f"RMSE = {rmse:.2f} €")
    print(f"R2   = {r2:.3f}")

    return 0


def main() -> int:
    # print("------ ERROR TESTS ------")

    # run_error_tests()

    # print("\n\n------ SUBJECT ------\n")

    args = parse_args()
    return evaluate(args.file, args.thetas)


if __name__ == "__main__":
    main()