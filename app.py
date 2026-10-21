import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="RGUKT RK Valley Wi-Fi Dashboard",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# FILE PATHS AND DATA
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "rgukt_rk_valley_wifi_dataset.csv"
MODEL_PATH = BASE_DIR / "wifi_speed_model.pkl"
BANNER_PATH = BASE_DIR / "rgukt_banner.avif"

FEATURES = [
    "Connected_Users",
    "Active_Devices",
    "Time_of_Day_Hour",
    "Day_of_Week",
    "Signal_Strength_dBm",
    "Available_Bandwidth_Mbps",
]

TARGET = "Download_Speed_Mbps"

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #174c72 100%);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 12px;
    }

    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        padding: 13px 16px;
        border-radius: 12px;
        width: 100%;
        font-size: 17px;
        font-weight: 600;
        transition: 0.2s;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.23);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: white;
    }

    .hero {
        padding: 48px 35px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        background:
            linear-gradient(90deg, rgba(8,30,55,0.88),
            rgba(8,30,55,0.45)),
            linear-gradient(135deg, #174c72, #3c91b8);
        background-size: cover;
        background-position: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #f0f7ff;
        font-size: 17px;
        margin-bottom: 0;
    }

    .section-title {
        color: #173b57;
        font-size: 25px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e4eaf1;
        box-shadow: 0 3px 12px rgba(25,50,75,0.05);
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-size: 16px;
        font-weight: 600;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 22px;
        border: 1px solid #e4eaf1;
        border-radius: 16px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add campus image as banner background when available.
if BANNER_PATH.exists():
    encoded_image = base64.b64encode(BANNER_PATH.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .hero {{
            background-image:
                linear-gradient(90deg, rgba(8,30,55,0.88),
                rgba(8,30,55,0.42)),
                url("data:image/jpeg;base64,{encoded_image}");
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# LOAD DATA AND MODEL SAFELY
# --------------------------------------------------
try:
    if not DATA_PATH.exists():
        st.error(
            "Dataset file not found. Add "
            "'rgukt_rk_valley_wifi_dataset.csv' "
            "to the same folder as app.py."
        )
        st.stop()

    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Add "
            "'wifi_speed_model.pkl' "
            "to the same folder as app.py."
        )
        st.stop()

    df = load_dataset(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))

except Exception as error:
    st.error(f"Unable to load the app files: {error}")
    st.stop()

missing_columns = [
    column for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📶 RGUKT RK Valley")
    st.markdown("### Wi-Fi Analytics")
    st.caption("Machine Learning Dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📊 Complete Dataset",
            "🚀 Predict Wi-Fi Speed",
            "🕘 Prediction History",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Project:** Wi-Fi Speed Prediction")
    st.markdown("**Model:** Machine Learning")
    st.markdown("**Campus:** RGUKT RK Valley")

# --------------------------------------------------
# SHARED FUNCTIONS
# --------------------------------------------------
def show_banner(subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>📶 RGUKT RK Valley Wi-Fi Dashboard</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def safe_number(series):
    return pd.to_numeric(series, errors="coerce")


# Prediction history stays available while this app session is open.
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------------
if page == "🏠 Dashboard":
    show_banner(
        "Monitor network activity, explore Wi-Fi performance, "
        "and predict download speed using Machine Learning."
    )

    total_records = len(df)
    avg_speed = safe_number(df[TARGET]).mean()
    max_speed = safe_number(df[TARGET]).max()
    avg_users = safe_number(df["Connected_Users"]).mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📁 Dataset Records", f"{total_records:,}")
    c2.metric(
        "⚡ Average Download Speed",
        f"{avg_speed:.2f} Mbps" if pd.notna(avg_speed) else "N/A",
    )
    c3.metric(
        "🚀 Highest Download Speed",
        f"{max_speed:.2f} Mbps" if pd.notna(max_speed) else "N/A",
    )
    c4.metric(
        "👥 Average Connected Users",
        f"{avg_users:.1f}" if pd.notna(avg_users) else "N/A",
    )

    st.divider()

    show_section("📈 Wi-Fi Performance Analytics")

    left, right = st.columns(2)

    # Speed distribution
    with left:
        st.subheader("Download Speed Distribution")

        speeds = safe_number(df[TARGET]).dropna()

        if not speeds.empty:
            counts, edges = __import__("numpy").histogram(
                speeds, bins=12
            )
            labels = [
                f"{edges[i]:.1f}–{edges[i + 1]:.1f}"
                for i in range(len(edges) - 1)
            ]
            distribution = pd.DataFrame(
                {"Speed Range (Mbps)": labels, "Records": counts}
            ).set_index("Speed Range (Mbps)")

            st.bar_chart(distribution)
        else:
            st.info("No valid download-speed values to display.")

    # Speed vs connected users
    with right:
        st.subheader("Speed vs Connected Users")

        scatter_data = df[
            ["Connected_Users", TARGET]
        ].copy()

        scatter_data["Connected_Users"] = safe_number(
            scatter_data["Connected_Users"]
        )
        scatter_data[TARGET] = safe_number(scatter_data[TARGET])
        scatter_data = scatter_data.dropna()

        if not scatter_data.empty:
            st.scatter_chart(
                scatter_data,
                x="Connected_Users",
                y=TARGET,
                x_label="Connected Users",
                y_label="Download Speed (Mbps)",
            )
        else:
            st.info("No valid records to display.")

    st.subheader("Average Download Speed by Hour")

    hourly = df[["Time_of_Day_Hour", TARGET]].copy()
    hourly["Time_of_Day_Hour"] = safe_number(
        hourly["Time_of_Day_Hour"]
    )
    hourly[TARGET] = safe_number(hourly[TARGET])
    hourly = hourly.dropna()

    if not hourly.empty:
        hourly_avg = (
            hourly.groupby("Time_of_Day_Hour")[TARGET]
            .mean()
            .sort_index()
        )
        hourly_avg.index = hourly_avg.index.astype(int).astype(str)
        hourly_avg.index.name = "Hour of Day"
        st.line_chart(hourly_avg)
    else:
        st.info("No valid hourly data to display.")

    st.caption(
        "Charts are based on the dataset in the repository. "
        "Predictions are estimates, not live Wi-Fi measurements."
    )

# --------------------------------------------------
# PAGE 2: COMPLETE DATASET
# --------------------------------------------------
elif page == "📊 Complete Dataset":
    show_banner("Explore, search, filter, and download the Wi-Fi dataset.")

    st.metric("Total Dataset Rows", f"{len(df):,}")
    st.metric("Total Dataset Columns", len(df.columns))

    show_section("🔎 Search and Filter Data")

    search_text = st.text_input(
        "Search across all columns",
        placeholder="Type a value, hour, day, or signal strength...",
    )

    filtered_df = df.copy()

    if search_text.strip():
        text_df = filtered_df.astype(str)
        matches = text_df.apply(
            lambda column: column.str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)
        filtered_df = filtered_df[matches]

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        if "Connected_Users" in filtered_df.columns:
            users_series = safe_number(filtered_df["Connected_Users"])
            valid_users = users_series.dropna()

            if not valid_users.empty:
                min_users = int(valid_users.min())
                max_users = int(valid_users.max())

                if min_users < max_users:
                    user_range = st.slider(
                        "Connected Users Range",
                        min_value=min_users,
                        max_value=max_users,
                        value=(min_users, max_users),
                    )
                    filtered_df = filtered_df[
                        safe_number(filtered_df["Connected_Users"])
                        .between(user_range[0], user_range[1])
                    ]

    with filter_col2:
        if "Time_of_Day_Hour" in filtered_df.columns:
            hours = safe_number(filtered_df["Time_of_Day_Hour"])
            valid_hours = hours.dropna()

            if not valid_hours.empty:
                min_hour = int(valid_hours.min())
                max_hour = int(valid_hours.max())

                if min_hour < max_hour:
                    hour_range = st.slider(
                        "Hour Range",
                        min_value=min_hour,
                        max_value=max_hour,
                        value=(min_hour, max_hour),
                    )
                    filtered_df = filtered_df[
                        safe_number(filtered_df["Time_of_Day_Hour"])
                        .between(hour_range[0], hour_range[1])
                    ]

    st.write(f"Showing **{len(filtered_df):,}** matching records.")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Displayed Data as CSV",
        data=csv_data,
        file_name="rgukt_wifi_dataset_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Clear the search box and widen the filters to display "
        "the entire dataset."
    )

# --------------------------------------------------
# PAGE 3: PREDICT WI-FI SPEED
# --------------------------------------------------
elif page == "🚀 Predict Wi-Fi Speed":
    show_banner(
        "Enter the network details below to estimate Wi-Fi download speed."
    )

    show_section("🧠 Enter Network Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            users = st.number_input(
                "👥 Connected Users",
                min_value=1,
                max_value=1000,
                value=25,
                step=1,
            )

            devices = st.number_input(
                "📱 Active Devices",
                min_value=1,
                max_value=2000,
                value=35,
                step=1,
            )

            hour = st.slider(
                "🕒 Time of Day",
                min_value=0,
                max_value=23,
                value=12,
            )

        with col2:
            day = st.selectbox(
                "📅 Day of Week",
                DAY_NAMES,
                index=0,
            )

            signal = st.slider(
                "📶 Wi-Fi Signal Strength (dBm)",
                min_value=-90,
                max_value=-30,
                value=-55,
            )

            bandwidth = st.selectbox(
                "🌐 Available Bandwidth (Mbps)",
                [100, 200, 300, 500],
                index=1,
            )

        submitted = st.form_submit_button(
            "🚀 Predict Wi-Fi Speed",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "Connected_Users": users,
                    "Active_Devices": devices,
                    "Time_of_Day_Hour": hour,
                    "Day_of_Week": DAY_NAMES.index(day),
                    "Signal_Strength_dBm": signal,
                    "Available_Bandwidth_Mbps": bandwidth,
                }
            ],
            columns=FEATURES,
        )

        try:
            prediction = float(model.predict(input_data)[0])

            if not pd.notna(prediction) or prediction < 0:
                st.error(
                    "The model returned an invalid prediction. "
                    "Please check your model and dataset."
                )
            else:
                st.success("✅ Prediction completed successfully!")
                st.balloons()

                st.metric(
                    "⚡ Predicted Download Speed",
                    f"{prediction:.2f} Mbps",
                )

                if prediction < 10:
                    st.warning(
                        "The predicted download speed is relatively low."
                    )
                elif prediction < 50:
                    st.info(
                        "The predicted download speed is in a moderate range."
                    )
                else:
                    st.success(
                        "The predicted download speed is in a higher range."
                    )

                st.caption(
                    "This is a Machine Learning estimate, not a live "
                    "internet speed test."
                )

                record = {
                    "Connected_Users": users,
                    "Active_Devices": devices,
                    "Time_of_Day_Hour": hour,
                    "Day_of_Week": day,
                    "Signal_Strength_dBm": signal,
                    "Available_Bandwidth_Mbps": bandwidth,
                    "Predicted_Download_Speed_Mbps": round(prediction, 2),
                }

                st.session_state.prediction_history.insert(0, record)

                with st.expander("View Input Details"):
                    st.dataframe(
                        input_data,
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# PAGE 4: PREDICTION HISTORY
# --------------------------------------------------
elif page == "🕘 Prediction History":
    show_banner("Review predictions made during your current app session.")

    show_section("🕘 Recent Predictions")

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        st.metric("Predictions This Session", len(history))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="wifi_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info(
            "No predictions yet. Open 'Predict Wi-Fi Speed' "
            "and make a prediction to see it here."
        )

# --------------------------------------------------
# PAGE 5: ABOUT PROJECT
# --------------------------------------------------
elif page == "ℹ️ About Project":
    show_banner("About the College Wi-Fi Speed Prediction project.")

    show_section("🎓 Project Overview")

    st.markdown(
        """
        This dashboard uses a Machine Learning model to estimate
        download speed from the supplied Wi-Fi network information.

        **Project features**
        - Dataset overview and visual analytics
        - Searchable and filterable dataset
        - CSV data export
        - Wi-Fi download-speed prediction
        - Session-based prediction history
        - RGUKT RK Valley campus banner support

        **Model input features**
        """
    )

    for feature in FEATURES:
        st.markdown(f"- `{feature}`")

    st.markdown(f"**Prediction target:** `{TARGET}`")

    st.info(
        "Predictions depend on the model and training data. "
        "They do not measure the current Wi-Fi connection directly."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:8px;">
        📶 RGUKT RK Valley | College Wi-Fi Speed Prediction
        <br>
        Machine Learning Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
