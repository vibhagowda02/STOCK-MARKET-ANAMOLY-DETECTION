import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# ======================
# Streamlit Config
# ======================
st.set_page_config(page_title="Stock Anomaly Detection (DL)", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #007bff;
    color:white;
    border-radius: 6px;
    padding: 6px 12px;
    font-size:14px;
    font-weight:600;
    width: auto;
}
div.stButton > button:hover {
    background-color: #0056b3;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Market Anomaly Detection ")

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("User Inputs")
tickers = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
    "META", "NVDA", "NFLX", "AMD", "INTC",
    "INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "HCLTECH.NS", "SBIN.NS"
]

symbol = st.sidebar.selectbox("Select Stock Symbol:", tickers)
# ensure start_date and end_date are date objects
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01").date())
end_date = st.sidebar.date_input("End Date", datetime.today().date())
load_data_button = st.sidebar.button("RUN DETECTION 🚀")

# ======================
# Load Data Function
# ======================
@st.cache_data(show_spinner=True)
def load_data(symbol, start, end):
    """
    Returns: (df, open_col)
    """
    # Prevent using a future date
    today = datetime.today().date()
    if end > today:
        st.warning(f"⚠️ End date {end} is in the future. Using today's date instead.")
        end = today

    # Convert to strings suitable for yfinance
    start_str = start.isoformat() if isinstance(start, date) else str(start)
    end_str = end.isoformat() if isinstance(end, date) else str(end)

    try:
        # download with symbol and correct date range
        data = yf.download(symbol, start=start_str, end=end_str, progress=False)
    except Exception as e:
        st.error(f"❌ Error downloading data for {symbol}: {e}")
        return pd.DataFrame(), None

    if data is None or data.empty:
        st.warning(f"⚠️ No data found for {symbol} between {start_str} and {end_str}. Try changing the range or ticker.")
        return pd.DataFrame(), None

    df = data.copy()

    # Flatten MultiIndex if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

    # Detect Close
    close_col = next((c for c in df.columns if "close" in c.lower()), None)
    if not close_col:
        st.error("No Close column found in downloaded data.")
        return pd.DataFrame(), None

    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    # Detect Open
    open_col = next((c for c in df.columns if "open" in c.lower()), None)

    # Detect Volume
    volume_col = next((c for c in df.columns if "volume" in c.lower()), None)
    df["Volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0) if volume_col else 0

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df, open_col

# ======================
# Main Process
# ======================
if load_data_button:
    st.write(f"🔍 Fetching data for **{symbol}** from **{start_date}** to **{end_date}**...")
    df, open_col = load_data(symbol, start_date, end_date)

    if df.empty:
        st.stop()

    # ======================
    # Feature Engineering
    # ======================
    df = df.sort_index()
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Return_MA10"] = df["Return"].rolling(10).mean()
    df["Return_STD10"] = df["Return"].rolling(10).std()
    df["Volatility20"] = df["Return"].rolling(20).std()
    df["Momentum5"] = (df["Close"] - df["Close"].shift(5)).fillna(0)
    df["RelativeVolume"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Z-score based anomalies
    Z_THRESHOLD = 3
    df["Return_Z"] = (df["Return"] - df["Return"].rolling(20).mean()) / df["Return"].rolling(20).std()
    df["Volume_Z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["Price_Anomaly"] = (df["Return_Z"].abs() > Z_THRESHOLD).fillna(False)
    df["Volume_Anomaly"] = (df["Volume_Z"].abs() > Z_THRESHOLD).fillna(False)

    # Pump & Dump detection
    df["Return_5dSum"] = df["Return"].rolling(5).sum()
    df["Next_Return"] = df["Return"].shift(-1)
    df["Pump_Dump"] = ((df["Return_5dSum"] > 0.05) & (df["Next_Return"] < -0.03)).fillna(False)
    df.drop(columns=["Return_5dSum", "Next_Return"], inplace=True)

    # Gap anomaly
    if open_col and open_col in df.columns:
        df["Gap"] = (df[open_col] - df["Close"].shift(1)) / df["Close"].shift(1)
        df["Gap_Anomaly"] = (df["Gap"].abs() > 0.03).fillna(False)
    else:
        df["Gap_Anomaly"] = False

    # Volatility anomaly
    df["Volatility_Anomaly"] = (
        df["Volatility20"] > df["Volatility20"].rolling(20).mean() + 2 * df["Volatility20"].rolling(20).std()
    ).fillna(False)

    # Momentum reversal
    df["Momentum_Anomaly"] = ((df["Momentum5"].shift(1) * df["Momentum5"] < 0) &
                              (df["Momentum5"].abs() > 0.02)).fillna(False)

    # Relative Volume anomaly
    df["RelativeVolume_Anomaly"] = ((df["RelativeVolume"] > 2) & (df["Return"].abs() < 0.005)).fillna(False)

    # ======================
    # Deep Learning — LSTM Autoencoder
    # ======================
    st.info("🧠 Training LSTM Autoencoder for anomaly detection...")

    features = ["Return", "Volume", "Volatility20", "Momentum5", "RelativeVolume"]
    train_df = df[features].dropna()

    if len(train_df) < 50:
        st.warning("Not enough data to train LSTM autoencoder reliably. Skipping DL anomaly detection.")
        df["DL_Anomaly"] = False
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train_df)

        TIME_STEPS = 20

        def create_sequences(data, steps=TIME_STEPS):
            X = []
            for i in range(len(data) - steps):
                X.append(data[i:i + steps])
            return np.array(X)

        X = create_sequences(scaled, TIME_STEPS)

        if X.size == 0:
            st.warning("Not enough rows to form sequences for LSTM. Skipping DL anomaly detection.")
            df["DL_Anomaly"] = False
        else:
            model = Sequential([
                LSTM(64, input_shape=(TIME_STEPS, X.shape[2]), return_sequences=False),
                RepeatVector(TIME_STEPS),
                LSTM(64, return_sequences=True),
                TimeDistributed(Dense(X.shape[2]))
            ])

            model.compile(optimizer="adam", loss="mse")
            model.fit(X, X, epochs=10, batch_size=32, verbose=0)

            reconstructions = model.predict(X, verbose=0)
            mse = np.mean((X - reconstructions) ** 2, axis=(1, 2))

            threshold = np.percentile(mse, 95)
            dl_flags = mse > threshold

            df["DL_Anomaly"] = False
            # align dl_flags with the tail of df (sequences start at TIME_STEPS index)
            start_idx = TIME_STEPS
            df.iloc[start_idx:start_idx + len(dl_flags), df.columns.get_loc("DL_Anomaly")] = dl_flags

    # ======================
    # Visualization Tabs
    # ======================
    plot_df = df.dropna()

    # Add anomaly timestamp column from DatetimeIndex
    plot_df["Anomaly_Timestamp"] = plot_df.index  # full date + time

    sns.set(style="whitegrid")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Anomalies", "Volume Anomalies", "Volatility & Returns", "Scatter Analysis", "Summary"
    ])

    # Tab 1 — Price
    with tab1:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(plot_df.index, plot_df["Close"], label="Close Price")
        anomaly_map = [
            ("Price_Anomaly", "Price Spike"),
            ("Pump_Dump", "Pump & Dump"),
            ("Gap_Anomaly", "Gap"),
            ("Momentum_Anomaly", "Momentum Reversal"),
            ("DL_Anomaly", "Deep Learning Anomaly")
        ]
        for col, label in anomaly_map:
            if col in plot_df.columns:
                xs = plot_df.loc[plot_df[col]].index
                ys = plot_df.loc[plot_df[col], "Close"]
                ax.scatter(xs, ys, label=label)
        ax.legend()
        ax.set_title(f"{symbol} — Price & DL Anomalies")
        fig.tight_layout()
        st.pyplot(fig)

    # Tab 2 — Volume
    with tab2:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(plot_df.index, plot_df["Volume"])
        if "Volume_Anomaly" in plot_df.columns:
            ax.scatter(plot_df.loc[plot_df["Volume_Anomaly"]].index,
                       plot_df.loc[plot_df["Volume_Anomaly"], "Volume"], label="Vol Spike")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    # Tab 3 — Volatility
    with tab3:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(plot_df.index, plot_df["Volatility20"], label="Volatility")
        if "Volatility_Anomaly" in plot_df.columns:
            ax.scatter(plot_df.loc[plot_df["Volatility_Anomaly"]].index,
                       plot_df.loc[plot_df["Volatility_Anomaly"], "Volatility20"], label="Vol Spike")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    # Tab 4 — Scatter
    with tab4:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x="Return", y="Volume", data=plot_df,
                        hue="DL_Anomaly", ax=ax)
        ax.set_title("DL Anomaly Clusters")
        fig.tight_layout()
        st.pyplot(fig)

    # Tab 5 — Summary
    with tab5:
        anomaly_cols = [
            "Price_Anomaly", "Volume_Anomaly", "Pump_Dump",
            "Gap_Anomaly", "Volatility_Anomaly", "Momentum_Anomaly",
            "RelativeVolume_Anomaly", "DL_Anomaly"
        ]
        # ensure columns exist
        for c in anomaly_cols:
            if c not in plot_df.columns:
                plot_df[c] = False

        plot_df["Anomaly_Count"] = plot_df[anomaly_cols].sum(axis=1)
        plot_df["Cumulative_Anomaly"] = plot_df["Anomaly_Count"].cumsum()

        st.subheader("Cumulative Anomalies")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(plot_df.index, plot_df["Cumulative_Anomaly"])
        fig.tight_layout()
        st.pyplot(fig)

        summary = {col: int(plot_df[col].sum()) for col in anomaly_cols}
        st.table(pd.DataFrame(summary, index=["Count"]))

        # ============= Anomalies with timestamps =============
        st.subheader("Anomalies with Date & Time")

        # Keep only rows where at least one anomaly occurred
        anomalies_only = plot_df[plot_df["Anomaly_Count"] > 0].copy()

        # Columns to display: timestamp, price, and anomaly flags
        cols_to_show = ["Anomaly_Timestamp", "Close"] + anomaly_cols
        st.dataframe(anomalies_only[cols_to_show])
