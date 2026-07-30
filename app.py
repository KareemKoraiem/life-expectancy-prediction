import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon="🌍",
    layout="wide"
)

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

h1{
    color:#2563EB;
    text-align:center;
}

h2,h3{
    color:#1E3A8A;
}

.stButton>button{
    width:100%;
    height:55px;
    background:#2563EB;
    color:white;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
    border:none;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white;
}

div[data-testid="metric-container"]{
    background:#EEF4FF;
    border:1px solid #C7D2FE;
    border-radius:15px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================
pipeline = joblib.load("life_expectancy_model.pkl")

# ==========================
# Sidebar
# ==========================

st.sidebar.title(" About")

st.sidebar.info("""
This application predicts Life Expectancy using a
Random Forest Regression model trained on the WHO dataset.
""")

st.sidebar.divider()

st.sidebar.metric("R² Score","0.9566")
st.sidebar.metric("MAE","1.34")
st.sidebar.metric("RMSE","1.94")

st.sidebar.divider()

st.sidebar.success("Developed using Streamlit & Scikit-learn")

# ==========================
# Header
# ==========================

st.title(" Life Expectancy Prediction")

st.markdown("""
Predict the **Life Expectancy** of a country using
health, vaccination and economic indicators.
""")

st.divider()

# ==========================
# Layout
# ==========================

left,right = st.columns(2)

# ==========================
# Left Column
# ==========================

with left:

    with st.expander(" General Information",expanded=True):

        year = st.slider(
            "Year",
            2000,
            2015,
            2015
        )

        status = st.selectbox(
            "Status",
            [
                "Developing",
                "Developed"
            ]
        )

    with st.expander("🩺 Health Indicators",expanded=True):

        infant_deaths = st.number_input(
            "Infant Deaths",
            min_value=0.0,
            value=20.0
        )

        alcohol = st.number_input(
            "Alcohol",
            min_value=0.0,
            value=5.0
        )

        percentage_expenditure = st.number_input(
            "Percentage Expenditure",
            min_value=0.0,
            value=500.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0
        )

        hiv = st.number_input(
            "HIV/AIDS",
            min_value=0.0,
            value=0.5
        )

        population = st.number_input(
            "Population",
            min_value=0.0,
            value=1000000.0,
            step=1000.0
        )

# ==========================
# Right Column
# ==========================

with right:

    with st.expander("💉 Vaccination",expanded=True):

        hepatitis_b = st.slider(
            "Hepatitis B",
            0,
            100,
            90
        )

        polio = st.slider(
            "Polio",
            0,
            100,
            90
        )

        diphtheria = st.slider(
            "Diphtheria",
            0,
            100,
            90
        )

        measles = st.number_input(
            "Measles",
            min_value=0.0,
            value=100.0
        )

    with st.expander("💰 Economy & Education",expanded=True):

        total_expenditure = st.number_input(
            "Total Expenditure",
            min_value=0.0,
            value=5.0
        )

        income = st.slider(
            "Income Composition",
            0.0,
            1.0,
            0.70
        )

        schooling = st.number_input(
            "Schooling",
            min_value=0.0,
            value=12.0
        )

        thinness = st.number_input(
            "Thinness (1-19 Years)",
            min_value=0.0,
            value=5.0
        )

st.divider()
# ==========================
# Predict Button
# ==========================

if st.button("🚀 Predict Life Expectancy", use_container_width=True):

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

        st.divider()

        st.success("Prediction Completed Successfully! 🎉")

        c1, c2, c3 = st.columns([1, 2, 1])

        with c2:

            st.metric(
                label=" Predicted Life Expectancy",
                value=f"{prediction:.2f} Years"
            )

        st.markdown("### 📋 Input Summary")

        summary = pd.DataFrame({
            "Feature": input_data.columns,
            "Value": input_data.iloc[0].values
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            f"""
### Interpretation

Based on the entered **health**, **vaccination**,
**education**, and **economic** indicators,

the estimated **Life Expectancy** is:

#  {prediction:.2f} Years
"""
        )

        if prediction >= 80:

            st.success("Excellent life expectancy.")

        elif prediction >= 70:

            st.success("Good life expectancy.")

        elif prediction >= 60:

            st.warning("Average life expectancy.")

        else:

            st.error("Low life expectancy.")

    except Exception as e:

        st.error("Prediction Failed ❌")
        st.exception(e)

# ==========================
# Footer
# ==========================

st.divider()

st.markdown(
"""
<div style='text-align:center'>

###  Life Expectancy Prediction

Developed with using

**Python • Streamlit • Scikit-learn • Random Forest**

---

Created by **Kareem Koraiem**

</div>
""",
unsafe_allow_html=True
)