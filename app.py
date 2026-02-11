import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Journal Analyzer", layout="wide")
st.title("R-Edge — Find Your Edge. Eliminate Your Losses.")

# ---- Sidebar: Calibration ----
st.sidebar.header("Execution Engine")
xau_price_divisor = st.sidebar.selectbox(
    "XAU price divisor (divide entry/exit by this)",
    options=[1, 10, 100, 1000],
    index=0
)

known_profit = st.sidebar.number_input(
    "Known MT4 profit for selected trade (optional)",
    value=0.0,
    step=0.01,
    help="Type the profit shown in MT4 for a specific trade, e.g. 28.75"
)

calib_row = st.sidebar.number_input(
    "Which row to use for calibration? (0 = first row)",
    min_value=0,
    value=0,
    step=1
)

show_debug = st.sidebar.checkbox("Show debug", value=False)

uploaded = st.file_uploader("Upload your journal CSV", type=["csv"])


def profit_factor(series: pd.Series) -> float:
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return dd.min()


def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ✅ risk column added
    required = ["date","symbol","direction","entry","exit","size","fees","setup","pip_value","risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()

    for col in ["entry","exit","size","fees","pip_value","risk"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["entry","exit","size","fees","pip_value","risk"]).copy()

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["direction"] = df["direction"].astype(str).str.lower().str.strip()
    df["setup"] = df["setup"].astype(str).str.strip()

    # Effective prices
    df["entry_eff"] = df["entry"]
    df["exit_eff"] = df["exit"]

    # Apply divisor for XAU only (if needed)
    xau_mask = df["symbol"].str.upper().str.contains("XAU", na=False)
    df.loc[xau_mask, "entry_eff"] = df.loc[xau_mask, "entry_eff"] / float(xau_price_divisor)
    df.loc[xau_mask, "exit_eff"]  = df.loc[xau_mask, "exit_eff"]  / float(xau_price_divisor)

    # Base PnL (uses CSV pip_value)
    df["gross_pnl_base"] = (df["exit_eff"] - df["entry_eff"]) * df["size"] * df["pip_value"]
    df.loc[df["direction"] == "short", "gross_pnl_base"] *= -1

    # Calibration multiplier to match MT4
    multiplier = 1.0
    if known_profit != 0.0 and len(df) > 0:
        r = int(calib_row)
        r = min(max(r, 0), len(df) - 1)
        base = float(df["gross_pnl_base"].iloc[r])
        if base != 0:
            multiplier = float(known_profit) / base

    df["gross_pnl"] = df["gross_pnl_base"] * multiplier
    df["net_pnl"] = df["gross_pnl"] - df["fees"]

    # R multiple (process metric)
    # If risk is <= 0, set R to NaN to avoid nonsense
    df["risk"] = df["risk"].abs()
    df.loc[df["risk"] <= 0, "risk"] = pd.NA
    df["R"] = df["net_pnl"] / df["risk"]

    df["is_win"] = df["net_pnl"] > 0
    df["day"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour

    df.attrs["calibration_multiplier"] = multiplier
    return df


if uploaded:
    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    mult = df.attrs.get("calibration_multiplier", 1.0)
    st.caption(f"Calibration multiplier applied: **{mult:.4f}**")

    if show_debug and len(df) > 0:
        st.subheader("Debug")
        r = min(max(int(calib_row), 0), len(df) - 1)
        row = df.iloc[r]
        st.write("Row used:", r)
        st.write("entry raw:", float(row["entry"]), "exit raw:", float(row["exit"]))
        st.write("entry eff:", float(row["entry_eff"]), "exit eff:", float(row["exit_eff"]))
        st.write("size:", float(row["size"]))
        st.write("pip_value:", float(row["pip_value"]))
        st.write("gross_pnl_base:", float(row["gross_pnl_base"]))
        st.write("multiplier:", float(mult))
        st.write("net_pnl:", float(row["net_pnl"]))
        st.write("risk:", float(row["risk"]))
        st.write("R:", float(row["R"]) if pd.notna(row["R"]) else None)

    # Filters
    colA, colB, colC = st.columns(3)
    with colA:
        symbols = st.multiselect("Symbol", sorted(df["symbol"].unique()), default=sorted(df["symbol"].unique()))
    with colB:
        setups = st.multiselect("Setup", sorted(df["setup"].unique()), default=sorted(df["setup"].unique()))
    with colC:
        directions = st.multiselect("Direction", sorted(df["direction"].unique()), default=sorted(df["direction"].unique()))

    f = df[
        df["symbol"].isin(symbols)
        & df["setup"].isin(setups)
        & df["direction"].isin(directions)
    ].copy()

    if f.empty:
        st.warning("No trades match filters.")
        st.stop()

    # ---- Core stats (money) ----
    total = f["net_pnl"].sum()
    trades = len(f)
    win_rate = f["is_win"].mean() * 100
    pf_money = profit_factor(f["net_pnl"])
    expectancy_money = f["net_pnl"].mean()

    # ---- Process stats (R) ----
    r_series = f["R"].dropna()
    avg_r = r_series.mean() if len(r_series) else 0.0
    expectancy_r = r_series.mean() if len(r_series) else 0.0
    pf_r = profit_factor(r_series) if len(r_series) else 0.0
    pct_over_2r = (r_series[r_series >= 2].count() / r_series.count() * 100) if len(r_series) else 0.0
    pct_under_minus1r = (r_series[r_series <= -1].count() / r_series.count() * 100) if len(r_series) else 0.0

    f = f.sort_values("date")
    f["equity"] = f["net_pnl"].cumsum()
    mdd = max_drawdown(f["equity"])

    # KPIs row 1 (money)
    st.subheader("Money Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Net P&L", f"{total:.2f}")
    k2.metric("Trades", trades)
    k3.metric("Win Rate", f"{win_rate:.1f}%")
    k4.metric("Profit Factor (money)", "∞" if pf_money == float("inf") else f"{pf_money:.2f}")
    k5.metric("Expectancy / trade (money)", f"{expectancy_money:.2f}")
    k6.metric("Max Drawdown (money)", f"{mdd:.2f}")

    # KPIs row 2 (R)
    st.subheader("Process Metrics (R)")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Avg R", f"{avg_r:.2f}")
    r2.metric("Expectancy (R)", f"{expectancy_r:.2f}")
    r3.metric("Profit Factor (R)", "∞" if pf_r == float("inf") else f"{pf_r:.2f}")
    r4.metric("% trades ≥ 2R", f"{pct_over_2r:.1f}%")
    r5.metric("% trades ≤ -1R", f"{pct_under_minus1r:.1f}%")

    st.divider()

    # Equity curve
    st.subheader("Equity Curve")
    fig = plt.figure()
    plt.plot(f["date"], f["equity"])
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Best/Worst
    left, right = st.columns(2)
    with left:
        st.subheader("Best Trades (by R)")
        st.dataframe(
            f.sort_values("R", ascending=False)[["date","symbol","setup","direction","net_pnl","risk","R"]].head(5),
            use_container_width=True
        )
    with right:
        st.subheader("Worst Trades (by R)")
        st.dataframe(
            f.sort_values("R", ascending=True)[["date","symbol","setup","direction","net_pnl","risk","R"]].head(5),
            use_container_width=True
        )

    st.divider()

    # Breakdowns
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("By Setup")
        by_setup = f.groupby("setup").agg(
            trades=("net_pnl","count"),
            pnl=("net_pnl","sum"),
            avg_R=("R","mean"),
            expectancy_R=("R","mean"),
            win_rate=("is_win","mean"),
        ).sort_values("pnl", ascending=False)
        by_setup["win_rate"] = (by_setup["win_rate"] * 100).round(1)
        by_setup["avg_R"] = by_setup["avg_R"].round(2)
        by_setup["expectancy_R"] = by_setup["expectancy_R"].round(2)
        st.dataframe(by_setup, use_container_width=True)

    with c2:
        st.subheader("By Day")
        by_day = f.groupby("day").agg(
            trades=("net_pnl","count"),
            pnl=("net_pnl","sum"),
            avg_R=("R","mean"),
            win_rate=("is_win","mean"),
        ).sort_values("pnl", ascending=False)
        by_day["win_rate"] = (by_day["win_rate"] * 100).round(1)
        by_day["avg_R"] = by_day["avg_R"].round(2)
        st.dataframe(by_day, use_container_width=True)

    with c3:
        st.subheader("By Hour")
        by_hour = f.groupby("hour").agg(
            trades=("net_pnl","count"),
            pnl=("net_pnl","sum"),
            avg_R=("R","mean"),
            win_rate=("is_win","mean"),
        ).sort_values("pnl", ascending=False)
        by_hour["win_rate"] = (by_hour["win_rate"] * 100).round(1)
        by_hour["avg_R"] = by_hour["avg_R"].round(2)
        st.dataframe(by_hour, use_container_width=True)

    st.divider()
    st.subheader("Raw Trades")
    st.dataframe(f, use_container_width=True)

else:
    st.info("Upload a CSV to begin.")
