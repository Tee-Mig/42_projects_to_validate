import argparse
import os
import pandas as pd
import tempfile

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

        zero_theta = os.path.join(tmpdir, "zero_thetas.txt")
        with open(zero_theta, "w", encoding="utf-8") as f:
            f.write("0,0")

        # csv files
        missing_csv = os.path.join(tmpdir, "missing.csv")

        missing_km_csv = os.path.join(tmpdir, "missing_km.csv")
        pd.DataFrame({"distance": [10000]}).to_csv(missing_km_csv, index=False)

        bad_km_csv = os.path.join(tmpdir, "bad_km.csv")
        pd.DataFrame({"km": ["mig", 20000]}).to_csv(bad_km_csv, index=False)

        valid_csv = os.path.join(tmpdir, "valid.csv")
        pd.DataFrame({"km": [10000, 20000, 30000]}).to_csv(valid_csv, index=False)

        negative_value_csv = os.path.join(tmpdir, "negative_value.csv")
        pd.DataFrame({"km": [-10000]}).to_csv(negative_value_csv, index=False)

        output_csv = os.path.join(tmpdir, "predictions.csv")

        # --- TESTS read_thetas ---
        print("\n----- read_thetas tests -----")

        theta_tests = [
            ("missing thetas file (defaults to 0,0)", missing_theta, False),
            ("empty thetas file", empty_theta, True),
            ("bad thetas format", bad_format_theta, True),
            ("non numeric thetas", bad_numeric_theta, True),
            ("valid thetas file", valid_theta, False),
        ]

        for name, theta_file, should_fail in theta_tests:
            print(f"\n--- Test: {name} ---")
            try:
                result = read_thetas(theta_file)

                if should_fail:
                    if result is None:
                        print("✅ expected error")
                    else:
                        print("❌ expected failure but got valid result")
                else:
                    if result is not None:
                        print("✅ success")
                    else:
                        print("❌ unexpected failure")

            except Exception as e:
                print(f"❌ unexpected exception: {e}")

        # --- TESTS predict_csv ---
        print("\n----- predict_csv tests -----")

        predict_tests = [
            ("missing csv file", missing_csv, valid_theta, True),
            ("csv missing km column", missing_km_csv, valid_theta, True),
            ("csv with non numeric km", bad_km_csv, valid_theta, True),
            ("valid csv", valid_csv, valid_theta, False),
            ("valid csv + zero thetas", valid_csv, zero_theta, False),
            ("km < 0", negative_value_csv, valid_theta, True),
        ]

        for name, csv_file, theta_file, should_fail in predict_tests:
            print(f"\n--- Test: {name} ---")

            thetas = read_thetas(theta_file)

            if thetas is None:
                print("❌ could not read thetas")
                continue

            theta0, theta1 = thetas

            try:
                result = predict_csv(csv_file, output_csv, theta0, theta1)

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
    parser = argparse.ArgumentParser(description="Predict car price from mileage")

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="CSV file containing km values (if no file: interactive mode)"
    )

    parser.add_argument(
        "--thetas",
        type=str,
        default="thetas.txt",
        help="Theta parameters file (default: thetas.txt)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file (default: predictions.csv)"
    )

    return parser.parse_args()


def read_thetas(theta_file: str) -> tuple[float, float] | None:
    if not os.path.exists(theta_file):
        print("Info: thetas file not found, defaulting to theta0=0, theta1=0")
        return (0.0, 0.0)

    try:
        with open(theta_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        parts = [p.strip() for p in content.split(",")]
        if len(parts) != 2:
            raise ValueError("Can't read theta0 and theta1")
        theta0 = float(parts[0])
        theta1 = float(parts[1])
        return theta0, theta1
    except Exception as e:
        print(f"Error reading thetas: {e}")
        return None


def predict_interactive(theta0: float, theta1: float) -> int:
    try:
        mileage_str = input("Enter mileage (km): ").strip()
        mileage = float(mileage_str)
    except Exception:
        print("Error: invalid mileage")
        return 1

    price = theta0 + (theta1 * mileage)
    print(f"Estimated price: {price:.2f}")
    return 0


def predict_csv(file_to_predict: str, output_file: str, theta0: float, theta1: float) -> int:
    if not os.path.exists(file_to_predict):
        print(f"Error: input file not found: {file_to_predict}")
        return 1

    if theta0 == 0.0 and theta1 == 0.0:
        print("Warning: thetas are 0 (model not trained). All predictions will be 0")

    try:
        df = pd.read_csv(file_to_predict)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1

    if "km" not in df.columns:
        print("Error: CSV must contain a 'km' column")
        return 1

    try:
        km_values = df["km"].astype(float)
    except ValueError:
        print("Error: column 'km' must contain numeric values")
        return 1
    
    if (km_values < 0).any():
        print("Error: column 'km' must contain only non-negative values")
        return 1


    prices = (theta0 + (theta1 * km_values)).round(2)

    result = pd.DataFrame({
        "km": km_values,
        "price": prices
    })

    try:
        result.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 1

    print(f"Predictions saved as {output_file}")
    return 0


def main() -> int:
    # print("-------- ERROR TESTS --------")

    # run_error_tests()

    # print("\n\n-------- SUBJECT --------\n")

    args = parse_args()

    thetas = read_thetas(args.thetas)
    if thetas is None:
        return 1
    theta0, theta1 = thetas

    if args.file is None:
        return predict_interactive(theta0, theta1)

    return predict_csv(args.file, args.output, theta0, theta1)


if __name__ == "__main__":
    main()