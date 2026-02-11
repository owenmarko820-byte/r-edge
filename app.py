import streamlit as st
import pandas as pd
import numpy as np


# ----------------------------
# Page + Brand
# ----------------------------
st.set_page_config(page_title="R-Edge", layout="wide")

st.title("R-Edge — Find Your Edge. Eliminate Your Losses.")
st.markdown(
    """
### Find Your Edge. Eliminate Costly Mistakes.
Built for traders who are done guessing and ready to trade like professionals.

**R-Edge reveals:**
- Your true expectancy
- Hidden execution leaks
- Setup-level edge
- Time-of-day performance inefficiencies

**Upload your journal. Get brutal clarity.**
"""
)

st.write("")  # spacing


# ----------------------------
# Sidebar (Control Centre)
# ----------------------------
with st.sidebar:
    st.header("Control Centre")
    st.caption("Upload your trades. Tune the engine. Find your edge.")

    with st.expander("Broker Sync", expanded=True):
        xau_price_divisor = st.selectbox(
            "Price scaling (divide entry/exit by)",
            options=[1, 10, 100, 1000],
            index=0,
            help="If your broker exports prices like 438581 instead of 4385.81, choose 100."
        )

        known_profit = st.number_input(
            "Broker P&L (optional)",
            value=0.0,
            step=0.01,
            help="If MT4 shows a different P&L than your CSV, enter it here to scale the CSV."
        )

        calib_row = st.number_input(
            "Calibration row (0 = first trade)",
            min_value=0,
            value=0,
            step=1,
            help="Choose which trade row to use for calibration when using Broker P&L."
        )

    show_debug = st.checkbox("Precision Diagnostics", value=False)


uploaded = st.file_uploader("Upload a CSV, then tune the engine to match your broker.", type=["csv"])


# ----------------------------
# Helpers
# ----------------------------
def profit_factor(series: pd.Series) -> float:
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    required = [
        "date", "symbol", "direction", "entry", "exit", "size", "fees",
        "setup", "pip_value", "risk"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Type conversions
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    df["setup"] = df["setup"].astype(str).str.strip().str.lower()

    # Make numeric
    numeric_cols = ["entry", "exit", "size", "fees", "pip_value", "risk"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing essentials
    df = df.dropna(subset=required).copy()

    # Apply price scaling
    df["entry_eff"] = df["entry"] / float(xau_price_divisor)
    df["exit_eff"] = df["exit"] / float(xau_price_divisor)

    # Gross P&L base (money)
    # direction: long -> (exit-entry); short -> (entry-exit)
    move = np.where(df["direction"] == "long",
                    df["exit_eff"] - df["entry_eff"],
                    df["entry_eff"] - df["exit_eff"])

    df["gross_pnl_base"] = move * df["size"] * df["pip_value"]
    df["fees"] = df["fees"].fillna(0.0)
    df["gross_pnl_base"] = df["gross_pnl_base"].fillna(0.0)

    # Optional calibration multiplier
    multiplier = 1.0
    base = safe_float(df["gross_pnl_base"].iloc[int(min(max(calib_row, 0), len(df) - 1))], 0.0)
    if known_profit != 0.0 and base != 0.0:
        multiplier = float(known_profit) / float(base)

    df["gross_pnl"] = df["gross_pnl_base"] * multiplier
    df["net_pnl"] = df["gross_pnl"] - df["fees"]

    # Risk / R metrics
    df["risk"] = df["risk"].abs()
    df.loc[df["risk"] <= 0, "risk"] = np.nan  # avoid nonsense
    df["R"] = df["net_pnl"] / df["risk"]

    df["is_win"] = df["net_pnl"] > 0
    df["day"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour

    df.attrs["calibration_multiplier"] = multiplier
    return df


# ----------------------------
# Main App (ONLY after upload)
# ----------------------------
if uploaded:
    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    mult = df.attrs.get("calibration_multiplier", 1.0)
    st.caption(f"Calibration multiplier applied: **{mult:.4f}**")

    # -----------------------------------
    # 🔥 Biggest Money Leak (TOTAL)
    # -----------------------------------
    st.subheader("🔥 Biggest Money Leak (Total)")
    leak_total = (
        df.groupby("setup")["net_pnl"]
        .sum()
        .sort_values()
    )

    if len(leak_total) > 0:
        worst_total_setup = leak_total.index[0]
        worst_total_value = float(leak_total.iloc[0])

        if worst_total_value < 0:
            st.error(
                f"📉 Largest Money Leak: **{worst_total_setup}** has lost you **{worst_total_value:.2f}** total."
            )
        else:
            st.success(
                f"✅ No losing setup detected. Weakest setup: **{worst_total_setup}** (total {worst_total_value:.2f})."
            )

    st.write("")

    # -----------------------------------
    # 🧠 Biggest Leak Detector (AVG)
    # -----------------------------------
    st.subheader("🧠 AI Performance Insight")

    leak_setup = (
        df.groupby("setup")["net_pnl"]
        .mean()
        .sort_values()
    )

    if len(leak_setup) > 0:
        worst_setup = leak_setup.index[0]
        worst_value = float(leak_setup.iloc[0])

        if worst_value < 0:
            st.error(
                f"🚨 Biggest Leak Detected: Your **{worst_setup}** setup is losing money "
                f"(avg **{worst_value:.2f}** per trade)."
            )
        else:
            st.success(
                f"✅ No losing setup detected. Weakest setup: **{worst_setup}** (avg {worst_value:.2f})."
            )

    st.write("")

    # -----------------------------------
    # Filters
    # -----------------------------------
    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)

    with c1:
        symbols = sorted(df["symbol"].unique().tolist())
        symbol_filter = st.multiselect("Symbol", symbols, default=symbols)

    with c2:
        setups = sorted(df["setup"].unique().tolist())
        setup_filter = st.multiselect("Setup", setups, default=setups)

    with c3:
        dirs = sorted(df["direction"].unique().tolist())
        dir_filter = st.multiselect("Direction", dirs, default=dirs)

    data = df[
        df["symbol"].isin(symbol_filter)
        & df["setup"].isin(setup_filter)
        & df["direction"].isin(dir_filter)
    ].copy()

    if len(data) == 0:
        st.warning("No rows match your filters.")
        st.stop()

    # -----------------------------------
    # Core metrics
    # -----------------------------------
    net = float(data["net_pnl"].sum())
    trades = int(len(data))
    win_rate = float((data["is_win"].mean()) * 100.0) if trades else 0.0
    pf_money = profit_factor(data["net_pnl"])
    expectancy_money = float(data["net_pnl"].mean())

    equity = data.sort_values("date")["net_pnl"].cumsum()
    mdd_money = max_drawdown(equity)

    avg_r = float(data["R"].mean()) if data["R"].notna().any() else 0.0
    pf_r = profit_factor(data["R"].dropna())
    exp_r = float(data["R"].dropna().mean()) if data["R"].notna().any() else 0.0
    pct_2r = float((data["R"] >= 2).mean() * 100.0) if "R" in data else 0.0
    pct_m1r = float((data["R"] <= -1).mean() * 100.0) if "R" in data else 0.0

    st.markdown("## Money Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Net P&L", f"{net:.2f}")
    m2.metric("Trades", f"{trades}")
    m3.metric("Win Rate", f"{win_rate:.1f}%")
    m4.metric("Profit Factor (money)", f"{pf_money:.2f}" if np.isfinite(pf_money) else "∞")
    m5.metric("Expectancy / trade (money)", f"{expectancy_money:.2f}")
    m6.metric("Max Drawdown (money)", f"{mdd_money:.2f}")

    st.write("")

    st.markdown("## Process Metrics (R)")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Avg R", f"{avg_r:.2f}")
    r2.metric("Expectancy (R)", f"{exp_r:.2f}")
    r3.metric("Profit Factor (R)", f"{pf_r:.2f}" if np.isfinite(pf_r) else "∞")
    r4.metric("% trades ≥ 2R", f"{pct_2r:.1f}%")
    r5.metric("% trades ≤ -1R", f"{pct_m1r:.1f}%")

    st.write("")

    # -----------------------------------
    # Best / Worst trades
    # -----------------------------------
    st.markdown("## Best / Worst Trades (by R)")
    left, right = st.columns(2)

    cols_show = ["date", "symbol", "setup", "direction", "net_pnl", "risk", "R"]
    best = data.sort_values("R", ascending=False).head(10)[cols_show]
    worst = data.sort_values("R", ascending=True).head(10)[cols_show]

    with left:
        st.write("### Best Trades (by R)")
        st.dataframe(best, use_container_width=True)

    with right:
        st.write("### Worst Trades (by R)")
        st.dataframe(worst, use_container_width=True)

    st.write("")

    # -----------------------------------
    # Breakdowns
    # -----------------------------------
    st.markdown("## Breakdowns")
    b1, b2, b3 = st.columns(3)

    by_setup = (
        data.groupby("setup")
        .agg(
            trades=("net_pnl", "count"),
            pnl=("net_pnl", "sum"),
            avg_R=("R", "mean"),
            expectancy_R=("R", "mean"),
            win_rate=("is_win", "mean"),
        )
        .reset_index()
    )
    by_setup["win_rate"] = (by_setup["win_rate"] * 100.0).round(2)

    by_day = (
        data.groupby("day")
        .agg(
            trades=("net_pnl", "count"),
            pnl=("net_pnl", "sum"),
            avg_R=("R", "mean"),
            win_rate=("is_win", "mean"),
        )
        .reset_index()
    )
    by_day["win_rate"] = (by_day["win_rate"] * 100.0).round(2)

    by_hour = (
        data.groupby("hour")
        .agg(
            trades=("net_pnl", "count"),
            pnl=("net_pnl", "sum"),
            avg_R=("R", "mean"),
            win_rate=("is_win", "mean"),
        )
        .reset_index()
        .sort_values("hour")
    )
    by_hour["win_rate"] = (by_hour["win_rate"] * 100.0).round(2)

    with b1:
        st.write("### By Setup")
        st.dataframe(by_setup, use_container_width=True)

    with b2:
        st.write("### By Day")
        st.dataframe(by_day, use_container_width=True)

    with b3:
        st.write("### By Hour")
        st.dataframe(by_hour, use_container_width=True)

    st.write("")

    # -----------------------------------
    # Raw trades
    # -----------------------------------
    st.markdown("## Raw Trades")
    st.dataframe(
        data.sort_values("date")[[
            "date", "symbol", "direction", "entry", "exit", "size", "fees",
            "setup", "pip_value", "risk", "entry_eff", "exit_eff",
            "gross_pnl_base", "gross_pnl", "net_pnl", "R", "is_win", "day", "hour"
        ]],
        use_container_width=True
    )

    # -----------------------------------
    # Diagnostics
    # -----------------------------------
    if show_debug:
        st.write("")
        st.subheader("Diagnostics")
        st.write("Rows:", len(df), "| Filtered rows:", len(data))
        st.write("Price divisor:", xau_price_divisor)
        st.write("Known P&L:", known_profit)
        st.write("Calibration row:", calib_row)
        st.write("Multiplier:", mult)

else:
    st.info("Upload a CSV to begin.")
