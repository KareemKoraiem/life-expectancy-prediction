import os
import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "## 🌍 Life Expectancy Prediction\n"
            "A Machine Learning regression app that predicts a country's "
            "life expectancy using WHO health, vaccination, education and "
            "economic indicators.\n\n"
            "**Model:** Random Forest  \n"
            "**R²:** 0.9566  \n"
            "**RMSE:** 1.94  \n"
            "**MAE:** 1.34"
        )
    }
)

# ==========================
# Load Model (safe loading)
# ==========================

MODEL_PATH = "life_expectancy_model.pkl"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pipeline = load_model(MODEL_PATH)

if pipeline is None:
    st.error(
        f"⚠️ Model file not found: `{MODEL_PATH}`. "
        "Please make sure the .pkl file is in the same folder as app.py."
    )
    st.stop()

# ==========================
# Minimal Custom CSS
# (relies on Streamlit's default dark theme for everything else,
# so it stays consistent across the whole app - sidebar included)
# ==========================

st.markdown("""
<style>

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: #2563EB;
    color: white;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background: #1D4ED8;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================

st.sidebar.header("About")

st.sidebar.info(
    "This application predicts Life Expectancy using a Random Forest "
    "Regression model trained on the WHO dataset."
)

st.sidebar.divider()

st.sidebar.metric("R² Score", "0.9566")
st.sidebar.metric("MAE", "1.34")
st.sidebar.metric("RMSE", "1.94")

st.sidebar.divider()

st.sidebar.success("Developed using Streamlit & Scikit-learn")

# ==========================
# Header
# ==========================

st.title("🌍 Life Expectancy Prediction")

st.write(
    "Predict the **Life Expectancy** of a country using health, "
    "vaccination and economic indicators."
)

st.divider()

# ==========================
# Input Sections
# ==========================

left, right = st.columns(2)

with left:

    with st.expander("General Information", expanded=True):

        year = st.slider("Year", 2000, 2015, 2015)

        status = st.selectbox("Development Status", ["Developing", "Developed"])

    with st.expander("🩺 Health Indicators", expanded=True):

        infant_deaths = st.number_input("Infant Deaths", min_value=0.0, value=20.0)

        alcohol = st.number_input("Alcohol Consumption", min_value=0.0, value=5.0)

        percentage_expenditure = st.number_input("Percentage Expenditure", min_value=0.0, value=500.0)

        bmi = st.number_input("BMI", min_value=0.0, value=25.0)

        hiv = st.number_input("HIV / AIDS", min_value=0.0, value=0.5)

        population = st.number_input("Population", min_value=0.0, value=1000000.0)

with right:

    with st.expander("💉 Vaccination", expanded=True):

        hepatitis_b = st.slider("Hepatitis B (%)", 0, 100, 90)

        polio = st.slider("Polio (%)", 0, 100, 90)

        diphtheria = st.slider("Diphtheria (%)", 0, 100, 90)

        measles = st.number_input("Measles Cases", min_value=0.0, value=100.0)

    with st.expander("💰 Economy & Education", expanded=True):

        total_expenditure = st.number_input("Total Expenditure", min_value=0.0, value=5.0)

        income = st.slider("Income Composition", 0.0, 1.0, 0.70)

        schooling = st.number_input("Schooling", min_value=0.0, value=12.0)

        thinness = st.number_input("Thinness (1-19 Years)", min_value=0.0, value=5.0)


st.write("")

predict = st.button("🚀 Predict Life Expectancy", use_container_width=True)

# ===================================
# Prediction
# ===================================

if predict:

    input_data = pd.DataFrame({
        "Year": [year],
        "Status": [status],
        "infant deaths": [infant_deaths],
        "Alcohol": [alcohol],
        "percentage expenditure": [percentage_expenditure],
        "Hepatitis B": [hepatitis_b],
        "Measles": [measles],
        "BMI": [bmi],
        "Polio": [polio],
        "Total expenditure": [total_expenditure],
        "Diphtheria": [diphtheria],
        "HIV/AIDS": [hiv],
        "Population": [population],
        "thinness  1-19 years": [thinness],
        "Income composition of resources": [income],
        "Schooling": [schooling]
    })

    try:
        prediction = pipeline.predict(input_data)[0]
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()

    st.write("")

    if prediction >= 80:
        status_text = "Excellent Life Expectancy"
        status_color = "#4ADE80"
        status_bg = "rgba(74, 222, 128, 0.15)"
        status_border = "rgba(74, 222, 128, 0.4)"
    elif prediction >= 70:
        status_text = "Good Life Expectancy"
        status_color = "#60A5FA"
        status_bg = "rgba(96, 165, 250, 0.15)"
        status_border = "rgba(96, 165, 250, 0.4)"
    elif prediction >= 60:
        status_text = "Average Life Expectancy"
        status_color = "#FBBF24"
        status_bg = "rgba(251, 191, 36, 0.15)"
        status_border = "rgba(251, 191, 36, 0.4)"
    else:
        status_text = "Low Life Expectancy"
        status_color = "#F87171"
        status_bg = "rgba(248, 113, 113, 0.15)"
        status_border = "rgba(248, 113, 113, 0.4)"

    st.markdown(
        f"""
        <div style="
            text-align:center;
            max-width:520px;
            margin:0 auto;
            padding:36px 28px;
            border-radius:20px;
            background:linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
            border:1px solid #334155;
            box-shadow:0px 8px 24px rgba(0,0,0,.25);
        ">
            <h3 style="margin-bottom:18px; color:#F8FAFC;">📊 Prediction Result</h3>
            <div style="color:#94A3B8; font-size:16px; margin-bottom:6px;">
                🌍 Life Expectancy
            </div>
            <div style="font-size:46px; font-weight:800; color:#60A5FA; margin-bottom:20px;">
                {prediction:.2f} Years
            </div>
            <div style="
                display:inline-block;
                padding:10px 24px;
                border-radius:10px;
                background:{status_bg};
                border:1px solid {status_border};
                color:{status_color};
                font-weight:700;
                font-size:16px;
            ">
                {status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
