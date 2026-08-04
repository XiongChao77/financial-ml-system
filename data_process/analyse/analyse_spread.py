import pandas as pd
import numpy as np
from pathlib import Path


def add_spread_and_fee_features(
    input_file: str | Path,
    output_file: str | Path | None = None,
    contract_size: float = 100.0,   # 1 volume = 100 XAU
    use_price: str = "open",        # "open" or "close"
):
    """
    Spread and approximate one-way commission from the bid/ask OHLC.

    spread_abs:
        ask - bid, in price units, e.g. USD / XAU for XAUUSD

    spread_pct:
        spread_abs / mid_price

    single_side_spread_abs:
        spread_abs / 2

    single_side_spread_pct:
        spread_pct / 2

    single_side_fee_per_volume:
        One-way spread cost, computed for 1 volume.
        With 1 volume = 100 XAU that gives:
        fee = spread / 2 * 100
    """

    input_file = Path(input_file)
    df = pd.read_csv(input_file)

    required_cols = [
        "open_bid", "high_bid", "low_bid", "close_bid", "volume_bid",
        "open_ask", "high_ask", "low_ask", "close_ask",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert to numbers
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # 1. OHLC mid
    # =========================
    df["open_mid"] = (df["open_bid"] + df["open_ask"]) / 2
    df["high_mid"] = (df["high_bid"] + df["high_ask"]) / 2
    df["low_mid"] = (df["low_bid"] + df["low_ask"]) / 2
    df["close_mid"] = (df["close_bid"] + df["close_ask"]) / 2

    # =========================
    # 2. OHLC spread
    # =========================
    df["spread_open"] = df["open_ask"] - df["open_bid"]
    df["spread_high"] = df["high_ask"] - df["high_bid"]
    df["spread_low"] = df["low_ask"] - df["low_bid"]
    df["spread_close"] = df["close_ask"] - df["close_bid"]

    # Relative spread
    eps = 1e-12

    df["spread_open_pct"] = df["spread_open"] / (df["open_mid"] + eps)
    df["spread_high_pct"] = df["spread_high"] / (df["high_mid"] + eps)
    df["spread_low_pct"] = df["spread_low"] / (df["low_mid"] + eps)
    df["spread_close_pct"] = df["spread_close"] / (df["close_mid"] + eps)

    # =========================
    # 3. Pick the spread used to approximate the commission
    # =========================
    if use_price == "open":
        spread_col = "spread_open"
        spread_pct_col = "spread_open_pct"
        mid_col = "open_mid"
    elif use_price == "close":
        spread_col = "spread_close"
        spread_pct_col = "spread_close_pct"
        mid_col = "close_mid"
    else:
        raise ValueError("use_price must be 'open' or 'close'")

    # One-way spread cost
    df["single_side_spread_abs"] = df[spread_col] / 2
    df["single_side_spread_pct"] = df[spread_pct_col] / 2

    # Approximate one-way spread cost per 1 volume
    # XAUUSD: 1 volume = 100 XAU
    df["single_side_spread_fee_per_volume"] = (
        df["single_side_spread_abs"] * contract_size
    )

    # Two-way spread cost
    df["round_turn_spread_abs"] = df[spread_col]
    df["round_turn_spread_pct"] = df[spread_pct_col]
    df["round_turn_spread_fee_per_volume"] = (
        df["round_turn_spread_abs"] * contract_size
    )

    # =========================
    # 4. Print the statistics
    # =========================
    stat_cols = [
        spread_col,
        spread_pct_col,
        "single_side_spread_abs",
        "single_side_spread_pct",
        "single_side_spread_fee_per_volume",
        "round_turn_spread_fee_per_volume",
    ]

    print("\n===== Spread / Fee Statistics =====")
    print(f"Rows: {len(df)}")
    print(f"Use spread from: {use_price}")
    print(f"Contract size: {contract_size}")

    summary = df[stat_cols].replace([np.inf, -np.inf], np.nan).describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )

    print(summary)

    # Output file
    if output_file is not None:
        output_file = Path(output_file)
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")

    return df, summary


if __name__ == "__main__":
    input_file = Path(r"/home/chao/work/QuantData/Forex/dukascopy/spot/XAUUSD_15m.csv")

    df, summary = add_spread_and_fee_features(
        input_file=input_file,
        output_file=None,
        contract_size=100.0,   # 1 volume = 100 XAU
        use_price="open",      # use open_bid/open_ask for the approximate opening spread
    )