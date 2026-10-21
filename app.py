import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="College Wi-Fi Speed Prediction",
    page_icon="📶",
    layout="wide"
)

DATA_FILE = "rgukt_rk_valley_wifi_dataset.csv"
MODEL_FILE = "wifi_speed_model.pkl"

FEATURES = [
    "Connected_Users",
    "Active_Devices",
    "Time_of_Day_Hour",
    "Day_of_Week",
    "Signal_Strength_dBm",
    "Available_Bandwidth_Mbps",
]
TARGET = "Download_Speed_Mbps"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: #f0f6ff;
    border: 1px solid #d8e6fa;
    padding: 16px;
    border-radius: 12px;
}
h1, h2, h3 {color: #174a7e;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.sidebar.title("📶 Wi-Fi Analytics")
st.sidebar.caption("RGUKT RK Valley")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Complete Dataset", "Predict Speed",
     "Prediction History", "About"]
)

st.title("📶 College Wi-Fi Speed Prediction")
st.caption("Network Analytics • Machine Learning • RGUKT RK Valley")

if not Path(DATA_FILE).exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"Missing column: {TARGET}")
    st.write("CSV columns:", list(df.columns))
    st.stop()

speed = pd.to_numeric(df[TARGET], errors="coerce").dropna()

# DASHBOARD
if page == "Dashboard":
    st.subheader("📊 Network Overview")
    a, b, c, d = st.columns(4)
    a.metric("Total Records", f"{len(df):,}")
    b.metric("Total Columns", len(df.columns))
    c.metric("Average Speed", f"{speed.mean():.2f} Mbps")
    d.metric("Maximum Speed", f"{speed.max():.2f} Mbps")

    left, right = st.columns(2)

    with left:
        st.subheader("Download Speed Distribution")
        st.histogram if False else None
        if not speed.empty:
            hist = pd.cut(speed, bins=10, duplicates="drop")
            counts = hist.value_counts().sort_index()
            counts.index = counts.index.astype(str)
            st.bar_chart(counts)

    with right:
        st.subheader("Speed vs Connected Users")
        if "Connected_Users" in df.columns:
            chart = pd.DataFrame({
                "Connected_Users": pd.to_numeric(
                    df["Connected_Users"], errors="coerce"),
                "Download_Speed_Mbps": speed.reindex(df.index)
            }).dropna()
            st.scatter_chart(
                chart,
                x="Connected_Users",
                y="Download_Speed_Mbps"
            )

    st.subheader("Average Speed by Hour")
    if "Time_of_Day_Hour" in df.columns:
        hourly = pd.DataFrame({
            "Hour": pd.to_numeric(
                df["Time_of_Day_Hour"], errors="coerce"),
            "Speed": pd.to_numeric(df[TARGET], errors="coerce")
        }).dropna()
        hourly = hourly.groupby("Hour")["Speed"].mean()
        st.line_chart(hourly)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# COMPLETE DATASET
elif page == "Complete Dataset":
    st.subheader("📋 Complete Wi-Fi Dataset")
    st.write(f"{len(df):,} records • {len(df.columns)} columns")

    search = st.text_input("🔎 Search all columns")
    filtered = df.copy()

    if search:
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, regex=False, na=False
            )
        ).any(axis=1)
        filtered = filtered[mask]

    with st.expander("Filters"):
        for col in [
            "Day_of_Week", "Time_of_Day_Hour", "Connected_Users"
        ]:
            if col in filtered.columns:
                values = sorted(
                    filtered[col].dropna().unique().tolist(),
                    key=str
                )
                selected = st.multiselect(
                    f"Filter {col}", values, key=f"filter_{col}"
                )
                if selected:
                    filtered = filtered[
                        filtered[col].isin(selected)
                    ]

    st.caption(f"Showing {len(filtered):,} rows")
    st.dataframe(
        filtered, use_container_width=True, height=520
    )

    st.download_button(
        "⬇️ Download Filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "wifi_dataset_filtered.csv",
        "text/csv"
    )
    st.download_button(
        "⬇️ Download Full Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "wifi_dataset.csv",
        "text/csv"
    )

# PREDICTION
elif page == "Predict Speed":
    st.subheader("🤖 Predict Wi-Fi Speed")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            users = st.number_input(
                "Connected Users", 1, 1000, 25
            )
            devices = st.number_input(
                "Active Devices", 1, 2000, 35
            )

        with c2:
            hour = st.slider("Time of Day", 0, 23, 12)
            days = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ]
            day = st.selectbox("Day of Week", days)

        with c3:
            signal = st.slider(
                "Signal Strength (dBm)", -90, -30, -55
            )
            bandwidth = st.selectbox(
                "Available Bandwidth (Mbps)",
                [100, 200, 300, 500], index=1
            )

        submit = st.form_submit_button(
            "🚀 Predict Wi-Fi Speed",
            use_container_width=True
        )

    if submit:
        try:
            model = load_model()
            input_df = pd.DataFrame([{
                "Connected_Users": users,
                "Active_Devices": devices,
                "Time_of_Day_Hour": hour,
                "Day_of_Week": days.index(day),
                "Signal_Strength_dBm": signal,
                "Available_Bandwidth_Mbps": bandwidth,
            }], columns=FEATURES)

            result = float(model.predict(input_df)[0])

            if not pd.notna(result) or result < 0:
                st.warning("The model returned an invalid speed.")
            else:
                st.success("Prediction completed!")
                st.metric(
                    "Predicted Download Speed",
                    f"{result:.2f} Mbps"
                )
                st.caption(
                    "ML estimate, not a live Wi-Fi speed test."
                )

                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    **input_df.iloc[0].to_dict(),
                    TARGET: round(result, 2)
                })

        except Exception as e:
            st.error("Prediction failed. Check model inputs.")
            st.exception(e)

# PREDICTION HISTORY
elif page == "Prediction History":
    st.subheader("📝 Prediction History")
    history = st.session_state.get("history", [])

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download History CSV",
            history_df.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv"
        )
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions in this session yet. "
            "Open Predict Speed to make a prediction."
        )

# ABOUT
elif page == "About":
    st.subheader("ℹ️ About This Project")
    st.write(
        "This application explores college Wi-Fi network data "
        "and estimates download speed using a trained ML model."
    )
    st.write(f"Dataset records: {len(df):,}")
    st.write("Model input features:")
    st.code("\n".join(FEATURES))
    st.caption(
        "Charts summarize the dataset. Predictions are estimates, "
        "not measurements of live network performance."
    )
