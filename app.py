import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# RGUKT RK Valley | College Wi-Fi Speed Prediction Dashboard
# Keep app.py, the CSV dataset, and model file in the same folder.
# ============================================================
st.set_page_config(
    page_title="College Wi-Fi Speed Prediction | RGUKT RK Valley",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "rgukt_rk_valley_wifi_dataset.csv"
MODEL_PATH = BASE_DIR / "wifi_speed_model.pkl"
BANNER_PATH = BASE_DIR / "rgukt_banner.avif"  # optional campus photo
TARGET = "Download_Speed_Mbps"
FEATURES = [
    "Connected_Users",
    "Active_Devices",
    "Time_of_Day_Hour",
    "Day_of_Week",
    "Signal_Strength_dBm",
    "Available_Bandwidth_Mbps",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Exact visual theme inspired by the preview: navy, royal blue, cyan, icy background.
NAVY = "#082B52"
NAVY_DARK = "#062447"
ROYAL = "#1677F8"
CYAN = "#19B9F2"
PALE = "#F3F8FF"
INK = "#103A78"
MUTED = "#607A9B"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', Arial, sans-serif;
    }}
    .stApp {{ background: {PALE}; color: {INK}; }}
    .block-container {{ max-width: 1600px; padding-top: 1.05rem; padding-bottom: 1.5rem; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY_DARK} 0%, {NAVY} 55%, #0B477A 100%);
        border-right: 1px solid #164b7a;
    }}
    [data-testid="stSidebar"] * {{ color: #F4F9FF; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap: 10px; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        box-sizing: border-box; width: 100%; min-height: 48px; padding: 12px 14px;
        border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
        background: rgba(255,255,255,.075); font-size: 15px; font-weight: 600;
        transition: background .15s ease;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(25,185,242,.23); border-color: rgba(25,185,242,.7);
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(90deg, {ROYAL}, #258DFB); border-color: #4EA4FF;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {{ color: white !important; }}
    .brand {{ padding: 6px 2px 14px; }}
    .brand-icon {{ font-size: 39px; line-height: 1; }}
    .brand-title {{ font-size: 19px; font-weight: 800; line-height: 1.35; color: white; }}
    .brand-sub {{ font-size: 12px; color: #B9D7F5; margin-top: 5px; }}
    .sidebar-bottom {{ margin-top: 22px; padding: 18px 8px 8px; border-top: 1px solid #2A5A82; color: #C5DDF4; font-size: 12px; line-height: 1.8; }}
    .hero {{
        position: relative; overflow: hidden; min-height: 150px; padding: 30px 32px;
        border-radius: 15px; margin: 0 0 18px 0; color: white;
        background: linear-gradient(100deg, rgba(4,35,72,.97), rgba(10,82,145,.83), rgba(16,119,197,.48)),
                    linear-gradient(120deg, #0A3566, #1677B8);
        background-position: center; background-size: cover;
        box-shadow: 0 6px 20px rgba(14,64,120,.12);
    }}
    .hero h1 {{ color: #fff !important; font-size: clamp(25px, 2.5vw, 36px); font-weight: 800; margin: 0 0 7px 0; letter-spacing: -.7px; }}
    .hero .hero-sub {{ color: #F2F8FF; font-size: 16px; margin: 0; }}
    .hero .hero-tag {{ color: #D3EEFF; font-size: 12px; margin-top: 9px; letter-spacing: .5px; }}
    .wifi-mark {{ float: left; font-size: 51px; margin-right: 20px; margin-top: 7px; }}
    .section-heading {{ font-size: 22px; font-weight: 800; color: #123F91; margin: 14px 0 10px; }}
    .panel {{ background: #fff; border: 1px solid #D5E5FA; border-radius: 13px; padding: 15px 17px; }}
    [data-testid="stMetric"] {{ background: white; border: 1px solid #D5E5FA; border-radius: 12px; padding: 16px 17px; box-shadow: 0 2px 8px rgba(20,74,140,.035); }}
    [data-testid="stMetricLabel"] {{ color: #174A8A; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: #082F6C; font-weight: 800; }}
    h1, h2, h3 {{ color: #123F91; }}
    div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] button {{
        border-radius: 10px; min-height: 43px; font-weight: 700; border: 1px solid #1677F8;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #D5E5FA; border-radius: 10px; overflow: hidden; }}
    div[data-testid="stForm"] {{ background: white; border: 1px solid #D5E5FA; border-radius: 14px; padding: 20px; }}
    .footer {{ margin-top: 18px; padding: 12px 16px; border-radius: 10px; background: #E7F1FF; color: #285C99; font-size: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Optional local campus banner. For a real campus photo, upload rgukt_banner.jpg to the repo.
if BANNER_PATH.exists():
    try:
        image_b64 = base64.b64encode(BANNER_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""<style>.hero {{background-image: linear-gradient(90deg,rgba(4,35,72,.94),rgba(4,55,105,.72),rgba(4,55,105,.28)), url('data:image/jpeg;base64,{image_b64}');}}</style>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

@st.cache_data
def load_dataset(path, mtime):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path, mtime):
    return joblib.load(path)

def numeric(series):
    return pd.to_numeric(series, errors="coerce")

def banner(subtitle):
    st.markdown(
        f"""<div class="hero"><div class="wifi-mark">📶</div><h1>College Wi-Fi Speed Prediction</h1><p class="hero-sub">RGUKT RK Valley</p><p class="hero-tag">{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )

def heading(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)

# Load real project files; don't silently substitute sample data.
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Upload it beside app.py in GitHub.")
    st.stop()
if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH.name}. Upload it beside app.py in GitHub.")
    st.stop()
try:
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Expected columns missing from the dataset: " + ", ".join(missing))
    st.write("Columns found:", list(df.columns))
    st.stop()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">📶</div><div class="brand-title">RGUKT RK Valley<br>Wi-Fi Analytics</div><div class="brand-sub">Machine Learning Dashboard</div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Dashboard", "Complete Dataset", "Predict Wi-Fi Speed", "Prediction History", "About Project"],
        format_func=lambda x: {
            "Dashboard": "🏠  Dashboard",
            "Complete Dataset": "🗄️  Complete Dataset",
            "Predict Wi-Fi Speed": "📡  Predict Wi-Fi Speed",
            "Prediction History": "🕘  Prediction History",
            "About Project": "ℹ️  About Project",
        }[x],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-bottom">📶<br><br>Project: Wi-Fi Speed Prediction<br>Model: Machine Learning<br>Campus: RGUKT RK Valley</div>',
        unsafe_allow_html=True,
    )

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------- DASHBOARD ---------------------------
if page == "Dashboard":
    banner("Smart Campus Network Analytics")
    speeds = numeric(df[TARGET]).dropna()
    users = numeric(df["Connected_Users"]).dropna()
    avg_speed = speeds.mean() if not speeds.empty else np.nan
    max_speed = speeds.max() if not speeds.empty else np.nan
    avg_users = users.mean() if not users.empty else np.nan

    a, b, c, d = st.columns(4)
    a.metric("🗄️  Dataset Records", f"{len(df):,}")
    b.metric("⚡  Average Download Speed", f"{avg_speed:.2f} Mbps" if pd.notna(avg_speed) else "N/A")
    c.metric("🚀  Peak Download Speed", f"{max_speed:.2f} Mbps" if pd.notna(max_speed) else "N/A")
    d.metric("👥  Average Connected Users", f"{avg_users:.1f}" if pd.notna(avg_users) else "N/A")

    heading("▥  Wi-Fi Performance Analytics")
    chart1, chart2, chart3 = st.columns([1.08, 1.08, 1.08])
    with chart1:
        st.markdown("**📶  Download Speed Distribution**")
        if not speeds.empty:
            counts, edges = np.histogram(speeds, bins=10)
            labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges)-1)]
            hist_df = pd.DataFrame({"Speed range (Mbps)": labels, "Records": counts}).set_index("Speed range (Mbps)")
            st.bar_chart(hist_df, height=260, color=ROYAL)
        else:
            st.info("No numeric speed data available.")
    with chart2:
        st.markdown("**🔵  Speed vs Connected Users**")
        scatter = pd.DataFrame({"Connected Users": numeric(df["Connected_Users"]), "Download Speed (Mbps)": numeric(df[TARGET])}).dropna()
        if not scatter.empty:
            st.scatter_chart(scatter, x="Connected Users", y="Download Speed (Mbps)", height=260, color=ROYAL)
        else:
            st.info("No data available for this chart.")
    with chart3:
        st.markdown("**🕒  Average Speed by Hour**")
        hourly = pd.DataFrame({"Hour": numeric(df["Time_of_Day_Hour"]), "Speed": numeric(df[TARGET])}).dropna()
        if not hourly.empty:
            hourly_avg = hourly.groupby("Hour")["Speed"].mean().sort_index()
            hourly_avg.index = hourly_avg.index.astype(int).astype(str)
            hourly_avg.index.name = "Hour of Day"
            st.line_chart(hourly_avg, height=260, color="#7457FF")
        else:
            st.info("No hourly data available.")

    heading("🗄️  Complete Dataset")
    st.caption("Preview of the first five records. Open Complete Dataset to search and view every row.")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    st.markdown('<div class="footer">ℹ️   Charts are based on the dataset in this repository. Predictions are estimates, not live Wi-Fi measurements. <span style="float:right">📶   Better Connectivity   |   Smarter Campus</span></div>', unsafe_allow_html=True)

# --------------------------- COMPLETE DATASET ---------------------------
elif page == "Complete Dataset":
    banner("Search, filter, inspect, and download the full Wi-Fi dataset")
    heading("🗄️  Complete Dataset")
    m1, m2 = st.columns(2)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Total Columns", len(df.columns))
    query = st.text_input("🔎 Search across all columns", placeholder="Search a value, day, signal strength...")
    filtered = df.copy()
    if query.strip():
        mask = filtered.astype(str).apply(lambda col: col.str.contains(query.strip(), case=False, regex=False, na=False)).any(axis=1)
        filtered = filtered.loc[mask]

    with st.expander("⚙️  Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            if "Day_of_Week" in filtered.columns:
                day_options = sorted(filtered["Day_of_Week"].dropna().unique().tolist(), key=str)
                selected_days = st.multiselect("Day of Week", day_options)
                if selected_days:
                    filtered = filtered[filtered["Day_of_Week"].isin(selected_days)]
        with f2:
            h = numeric(filtered["Time_of_Day_Hour"]).dropna()
            if not h.empty and h.min() < h.max():
                hrange = st.slider("Hour range", int(h.min()), int(h.max()), (int(h.min()), int(h.max())))
                filtered = filtered[numeric(filtered["Time_of_Day_Hour"]).between(*hrange)]
        with f3:
            u = numeric(filtered["Connected_Users"]).dropna()
            if not u.empty and u.min() < u.max():
                urange = st.slider("Connected users range", int(u.min()), int(u.max()), (int(u.min()), int(u.max())))
                filtered = filtered[numeric(filtered["Connected_Users"]).between(*urange)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, height=520, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️  Download Filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "rgukt_wifi_filtered.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️  Download Full Dataset CSV", df.to_csv(index=False).encode("utf-8"), "rgukt_wifi_full_dataset.csv", "text/csv", use_container_width=True)

# --------------------------- PREDICT ---------------------------
elif page == "Predict Wi-Fi Speed":
    banner("Enter network conditions to estimate download speed")
    heading("📡  Wi-Fi Speed Prediction")
    st.write("Complete all six inputs and click Predict. The result is an ML estimate, not a live speed test.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            connected = st.number_input("👥 Connected Users", min_value=1, max_value=1000, value=25, step=1)
            devices = st.number_input("📱 Active Devices", min_value=1, max_value=2000, value=35, step=1)
            hour = st.slider("🕒 Time of Day (hour)", 0, 23, 12)
        with right:
            day_name = st.selectbox("📅 Day of Week", DAYS, index=0)
            signal = st.slider("📶 Signal Strength (dBm)", -90, -30, -55)
            bandwidth = st.selectbox("🌐 Available Bandwidth (Mbps)", [100, 200, 300, 500], index=1)
        submitted = st.form_submit_button("📶  Predict Wi-Fi Speed", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Connected_Users": connected,
            "Active_Devices": devices,
            "Time_of_Day_Hour": hour,
            "Day_of_Week": DAYS.index(day_name),
            "Signal_Strength_dBm": signal,
            "Available_Bandwidth_Mbps": bandwidth,
        }], columns=FEATURES)
        try:
            result = float(model.predict(input_df)[0])
            if not np.isfinite(result) or result < 0:
                st.error("The model returned an invalid speed. Check the model and training data.")
            else:
                st.success("🎉 Prediction Completed Successfully!")
                st.balloons()
                st.metric("📶 Predicted Download Speed", f"{result:.2f} Mbps")
                st.progress(min(max(result / max(float(speeds.max()) if not speeds.empty else 100, 1), 0.0), 1.0))
                st.caption("Prediction output is estimated from the trained model; it does not measure current network speed.")
                st.session_state.prediction_history.insert(0, {
                    "Connected_Users": connected,
                    "Active_Devices": devices,
                    "Time_of_Day_Hour": hour,
                    "Day_of_Week": day_name,
                    "Signal_Strength_dBm": signal,
                    "Available_Bandwidth_Mbps": bandwidth,
                    "Predicted_Download_Speed_Mbps": round(result, 2),
                })
                with st.expander("View prediction inputs"):
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error("Prediction failed. The model may expect different feature names or input types.")
            st.exception(exc)

# --------------------------- HISTORY ---------------------------
elif page == "Prediction History":
    banner("Review predictions made in this active app session")
    heading("🕘  Prediction History")
    history = st.session_state.prediction_history
    if history:
        hist_df = pd.DataFrame(history)
        st.metric("Predictions This Session", len(hist_df))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️  Download Prediction History", hist_df.to_csv(index=False).encode("utf-8"), "rgukt_prediction_history.csv", "text/csv")
        if st.button("🗑️  Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Open Predict Wi-Fi Speed and run a prediction first.")
        st.caption("History is session-based and resets when the app session ends.")

# --------------------------- ABOUT ---------------------------
else:
    banner("Smart Campus Network Analytics")
    heading("ℹ️  About This Project")
    st.markdown("""
    **College Wi-Fi Speed Prediction** estimates download speed from Wi-Fi network conditions using a trained machine-learning model.

    **Dashboard features**
    - 📊 Dataset summary and performance charts
    - 🗄️ Searchable complete dataset and CSV exports
    - 📶 Prediction form using the six model input fields
    - 🎉 Prediction completion animation
    - 🕘 Session-based prediction history
    """)
    st.markdown("**Model input features**")
    st.code("\n".join(FEATURES), language="text")
    st.markdown(f"**Prediction target:** `{TARGET}`")
    st.info("Keep the trained model and dataset files beside app.py in the GitHub repository.")

st.markdown('<div class="footer">📶   RGUKT RK Valley   |   College Wi-Fi Speed Prediction   |   Machine Learning Dashboard</div>', unsafe_allow_html=True)import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# RGUKT RK Valley | College Wi-Fi Speed Prediction Dashboard
# Keep app.py, the CSV dataset, and model file in the same folder.
# ============================================================
st.set_page_config(
    page_title="College Wi-Fi Speed Prediction | RGUKT RK Valley",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "rgukt_rk_valley_wifi_dataset.csv"
MODEL_PATH = BASE_DIR / "wifi_speed_model.pkl"
BANNER_PATH = BASE_DIR / "rgukt_banner.jpg"  # optional campus photo
TARGET = "Download_Speed_Mbps"
FEATURES = [
    "Connected_Users",
    "Active_Devices",
    "Time_of_Day_Hour",
    "Day_of_Week",
    "Signal_Strength_dBm",
    "Available_Bandwidth_Mbps",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Exact visual theme inspired by the preview: navy, royal blue, cyan, icy background.
NAVY = "#082B52"
NAVY_DARK = "#062447"
ROYAL = "#1677F8"
CYAN = "#19B9F2"
PALE = "#F3F8FF"
INK = "#103A78"
MUTED = "#607A9B"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', Arial, sans-serif;
    }}
    .stApp {{ background: {PALE}; color: {INK}; }}
    .block-container {{ max-width: 1600px; padding-top: 1.05rem; padding-bottom: 1.5rem; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY_DARK} 0%, {NAVY} 55%, #0B477A 100%);
        border-right: 1px solid #164b7a;
    }}
    [data-testid="stSidebar"] * {{ color: #F4F9FF; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap: 10px; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        box-sizing: border-box; width: 100%; min-height: 48px; padding: 12px 14px;
        border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
        background: rgba(255,255,255,.075); font-size: 15px; font-weight: 600;
        transition: background .15s ease;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(25,185,242,.23); border-color: rgba(25,185,242,.7);
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(90deg, {ROYAL}, #258DFB); border-color: #4EA4FF;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {{ color: white !important; }}
    .brand {{ padding: 6px 2px 14px; }}
    .brand-icon {{ font-size: 39px; line-height: 1; }}
    .brand-title {{ font-size: 19px; font-weight: 800; line-height: 1.35; color: white; }}
    .brand-sub {{ font-size: 12px; color: #B9D7F5; margin-top: 5px; }}
    .sidebar-bottom {{ margin-top: 22px; padding: 18px 8px 8px; border-top: 1px solid #2A5A82; color: #C5DDF4; font-size: 12px; line-height: 1.8; }}
    .hero {{
        position: relative; overflow: hidden; min-height: 150px; padding: 30px 32px;
        border-radius: 15px; margin: 0 0 18px 0; color: white;
        background: linear-gradient(100deg, rgba(4,35,72,.97), rgba(10,82,145,.83), rgba(16,119,197,.48)),
                    linear-gradient(120deg, #0A3566, #1677B8);
        background-position: center; background-size: cover;
        box-shadow: 0 6px 20px rgba(14,64,120,.12);
    }}
    .hero h1 {{ color: #fff !important; font-size: clamp(25px, 2.5vw, 36px); font-weight: 800; margin: 0 0 7px 0; letter-spacing: -.7px; }}
    .hero .hero-sub {{ color: #F2F8FF; font-size: 16px; margin: 0; }}
    .hero .hero-tag {{ color: #D3EEFF; font-size: 12px; margin-top: 9px; letter-spacing: .5px; }}
    .wifi-mark {{ float: left; font-size: 51px; margin-right: 20px; margin-top: 7px; }}
    .section-heading {{ font-size: 22px; font-weight: 800; color: #123F91; margin: 14px 0 10px; }}
    .panel {{ background: #fff; border: 1px solid #D5E5FA; border-radius: 13px; padding: 15px 17px; }}
    [data-testid="stMetric"] {{ background: white; border: 1px solid #D5E5FA; border-radius: 12px; padding: 16px 17px; box-shadow: 0 2px 8px rgba(20,74,140,.035); }}
    [data-testid="stMetricLabel"] {{ color: #174A8A; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: #082F6C; font-weight: 800; }}
    h1, h2, h3 {{ color: #123F91; }}
    div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] button {{
        border-radius: 10px; min-height: 43px; font-weight: 700; border: 1px solid #1677F8;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #D5E5FA; border-radius: 10px; overflow: hidden; }}
    div[data-testid="stForm"] {{ background: white; border: 1px solid #D5E5FA; border-radius: 14px; padding: 20px; }}
    .footer {{ margin-top: 18px; padding: 12px 16px; border-radius: 10px; background: #E7F1FF; color: #285C99; font-size: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Optional local campus banner. For a real campus photo, upload rgukt_banner.jpg to the repo.
if BANNER_PATH.exists():
    try:
        image_b64 = base64.b64encode(BANNER_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""<style>.hero {{background-image: linear-gradient(90deg,rgba(4,35,72,.94),rgba(4,55,105,.72),rgba(4,55,105,.28)), url('data:image/jpeg;base64,{image_b64}');}}</style>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

@st.cache_data
def load_dataset(path, mtime):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path, mtime):
    return joblib.load(path)

def numeric(series):
    return pd.to_numeric(series, errors="coerce")

def banner(subtitle):
    st.markdown(
        f"""<div class="hero"><div class="wifi-mark">📶</div><h1>College Wi-Fi Speed Prediction</h1><p class="hero-sub">RGUKT RK Valley</p><p class="hero-tag">{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )

def heading(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)

# Load real project files; don't silently substitute sample data.
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Upload it beside app.py in GitHub.")
    st.stop()
if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH.name}. Upload it beside app.py in GitHub.")
    st.stop()
try:
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Expected columns missing from the dataset: " + ", ".join(missing))
    st.write("Columns found:", list(df.columns))
    st.stop()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">📶</div><div class="brand-title">RGUKT RK Valley<br>Wi-Fi Analytics</div><div class="brand-sub">Machine Learning Dashboard</div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Dashboard", "Complete Dataset", "Predict Wi-Fi Speed", "Prediction History", "About Project"],
        format_func=lambda x: {
            "Dashboard": "🏠  Dashboard",
            "Complete Dataset": "🗄️  Complete Dataset",
            "Predict Wi-Fi Speed": "📡  Predict Wi-Fi Speed",
            "Prediction History": "🕘  Prediction History",
            "About Project": "ℹ️  About Project",
        }[x],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-bottom">📶<br><br>Project: Wi-Fi Speed Prediction<br>Model: Machine Learning<br>Campus: RGUKT RK Valley</div>',
        unsafe_allow_html=True,
    )

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------- DASHBOARD ---------------------------
if page == "Dashboard":
    banner("Smart Campus Network Analytics")
    speeds = numeric(df[TARGET]).dropna()
    users = numeric(df["Connected_Users"]).dropna()
    avg_speed = speeds.mean() if not speeds.empty else np.nan
    max_speed = speeds.max() if not speeds.empty else np.nan
    avg_users = users.mean() if not users.empty else np.nan

    a, b, c, d = st.columns(4)
    a.metric("🗄️  Dataset Records", f"{len(df):,}")
    b.metric("⚡  Average Download Speed", f"{avg_speed:.2f} Mbps" if pd.notna(avg_speed) else "N/A")
    c.metric("🚀  Peak Download Speed", f"{max_speed:.2f} Mbps" if pd.notna(max_speed) else "N/A")
    d.metric("👥  Average Connected Users", f"{avg_users:.1f}" if pd.notna(avg_users) else "N/A")

    heading("▥  Wi-Fi Performance Analytics")
    chart1, chart2, chart3 = st.columns([1.08, 1.08, 1.08])
    with chart1:
        st.markdown("**📶  Download Speed Distribution**")
        if not speeds.empty:
            counts, edges = np.histogram(speeds, bins=10)
            labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges)-1)]
            hist_df = pd.DataFrame({"Speed range (Mbps)": labels, "Records": counts}).set_index("Speed range (Mbps)")
            st.bar_chart(hist_df, height=260, color=ROYAL)
        else:
            st.info("No numeric speed data available.")
    with chart2:
        st.markdown("**🔵  Speed vs Connected Users**")
        scatter = pd.DataFrame({"Connected Users": numeric(df["Connected_Users"]), "Download Speed (Mbps)": numeric(df[TARGET])}).dropna()
        if not scatter.empty:
            st.scatter_chart(scatter, x="Connected Users", y="Download Speed (Mbps)", height=260, color=ROYAL)
        else:
            st.info("No data available for this chart.")
    with chart3:
        st.markdown("**🕒  Average Speed by Hour**")
        hourly = pd.DataFrame({"Hour": numeric(df["Time_of_Day_Hour"]), "Speed": numeric(df[TARGET])}).dropna()
        if not hourly.empty:
            hourly_avg = hourly.groupby("Hour")["Speed"].mean().sort_index()
            hourly_avg.index = hourly_avg.index.astype(int).astype(str)
            hourly_avg.index.name = "Hour of Day"
            st.line_chart(hourly_avg, height=260, color="#7457FF")
        else:
            st.info("No hourly data available.")

    heading("🗄️  Complete Dataset")
    st.caption("Preview of the first five records. Open Complete Dataset to search and view every row.")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    st.markdown('<div class="footer">ℹ️   Charts are based on the dataset in this repository. Predictions are estimates, not live Wi-Fi measurements. <span style="float:right">📶   Better Connectivity   |   Smarter Campus</span></div>', unsafe_allow_html=True)

# --------------------------- COMPLETE DATASET ---------------------------
elif page == "Complete Dataset":
    banner("Search, filter, inspect, and download the full Wi-Fi dataset")
    heading("🗄️  Complete Dataset")
    m1, m2 = st.columns(2)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Total Columns", len(df.columns))
    query = st.text_input("🔎 Search across all columns", placeholder="Search a value, day, signal strength...")
    filtered = df.copy()
    if query.strip():
        mask = filtered.astype(str).apply(lambda col: col.str.contains(query.strip(), case=False, regex=False, na=False)).any(axis=1)
        filtered = filtered.loc[mask]

    with st.expander("⚙️  Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            if "Day_of_Week" in filtered.columns:
                day_options = sorted(filtered["Day_of_Week"].dropna().unique().tolist(), key=str)
                selected_days = st.multiselect("Day of Week", day_options)
                if selected_days:
                    filtered = filtered[filtered["Day_of_Week"].isin(selected_days)]
        with f2:
            h = numeric(filtered["Time_of_Day_Hour"]).dropna()
            if not h.empty and h.min() < h.max():
                hrange = st.slider("Hour range", int(h.min()), int(h.max()), (int(h.min()), int(h.max())))
                filtered = filtered[numeric(filtered["Time_of_Day_Hour"]).between(*hrange)]
        with f3:
            u = numeric(filtered["Connected_Users"]).dropna()
            if not u.empty and u.min() < u.max():
                urange = st.slider("Connected users range", int(u.min()), int(u.max()), (int(u.min()), int(u.max())))
                filtered = filtered[numeric(filtered["Connected_Users"]).between(*urange)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, height=520, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️  Download Filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "rgukt_wifi_filtered.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️  Download Full Dataset CSV", df.to_csv(index=False).encode("utf-8"), "rgukt_wifi_full_dataset.csv", "text/csv", use_container_width=True)

# --------------------------- PREDICT ---------------------------
elif page == "Predict Wi-Fi Speed":
    banner("Enter network conditions to estimate download speed")
    heading("📡  Wi-Fi Speed Prediction")
    st.write("Complete all six inputs and click Predict. The result is an ML estimate, not a live speed test.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            connected = st.number_input("👥 Connected Users", min_value=1, max_value=1000, value=25, step=1)
            devices = st.number_input("📱 Active Devices", min_value=1, max_value=2000, value=35, step=1)
            hour = st.slider("🕒 Time of Day (hour)", 0, 23, 12)
        with right:
            day_name = st.selectbox("📅 Day of Week", DAYS, index=0)
            signal = st.slider("📶 Signal Strength (dBm)", -90, -30, -55)
            bandwidth = st.selectbox("🌐 Available Bandwidth (Mbps)", [100, 200, 300, 500], index=1)
        submitted = st.form_submit_button("📶  Predict Wi-Fi Speed", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Connected_Users": connected,
            "Active_Devices": devices,
            "Time_of_Day_Hour": hour,
            "Day_of_Week": DAYS.index(day_name),
            "Signal_Strength_dBm": signal,
            "Available_Bandwidth_Mbps": bandwidth,
        }], columns=FEATURES)
        try:
            result = float(model.predict(input_df)[0])
            if not np.isfinite(result) or result < 0:
                st.error("The model returned an invalid speed. Check the model and training data.")
            else:
                st.success("🎉 Prediction Completed Successfully!")
                st.balloons()
                st.metric("📶 Predicted Download Speed", f"{result:.2f} Mbps")
                st.progress(min(max(result / max(float(speeds.max()) if not speeds.empty else 100, 1), 0.0), 1.0))
                st.caption("Prediction output is estimated from the trained model; it does not measure current network speed.")
                st.session_state.prediction_history.insert(0, {
                    "Connected_Users": connected,
                    "Active_Devices": devices,
                    "Time_of_Day_Hour": hour,
                    "Day_of_Week": day_name,
                    "Signal_Strength_dBm": signal,
                    "Available_Bandwidth_Mbps": bandwidth,
                    "Predicted_Download_Speed_Mbps": round(result, 2),
                })
                with st.expander("View prediction inputs"):
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error("Prediction failed. The model may expect different feature names or input types.")
            st.exception(exc)

# --------------------------- HISTORY ---------------------------
elif page == "Prediction History":
    banner("Review predictions made in this active app session")
    heading("🕘  Prediction History")
    history = st.session_state.prediction_history
    if history:
        hist_df = pd.DataFrame(history)
        st.metric("Predictions This Session", len(hist_df))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️  Download Prediction History", hist_df.to_csv(index=False).encode("utf-8"), "rgukt_prediction_history.csv", "text/csv")
        if st.button("🗑️  Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Open Predict Wi-Fi Speed and run a prediction first.")
        st.caption("History is session-based and resets when the app session ends.")

# --------------------------- ABOUT ---------------------------
else:
    banner("Smart Campus Network Analytics")
    heading("ℹ️  About This Project")
    st.markdown("""
    **College Wi-Fi Speed Prediction** estimates download speed from Wi-Fi network conditions using a trained machine-learning model.

    **Dashboard features**
    - 📊 Dataset summary and performance charts
    - 🗄️ Searchable complete dataset and CSV exports
    - 📶 Prediction form using the six model input fields
    - 🎉 Prediction completion animation
    - 🕘 Session-based prediction history
    """)
    st.markdown("**Model input features**")
    st.code("\n".join(FEATURES), language="text")
    st.markdown(f"**Prediction target:** `{TARGET}`")
    st.info("Keep the trained model and dataset files beside app.py in the GitHub repository.")

st.markdown(
    """
    <div class="footer">
        📶 RGUKT RK Valley |
        College Wi-Fi Speed Prediction |
        Machine Learning Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
