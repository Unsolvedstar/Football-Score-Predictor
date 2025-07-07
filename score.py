import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("match_outcome_predictor.pkl")
le_team = joblib.load("team_encoder.pkl")
le_result = joblib.load("result_encoder.pkl")

teams = le_team.classes_

# App title with football emoji
st.set_page_config(page_title="⚽ EPL Match Predictor", page_icon="⚽")
st.title("⚽ Premier League Match Outcome Predictor")
st.caption("Predict who might win in your next Premier League match!")

# Add football image/banner (optional)
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stButton>button {
        background-color: #0254a0;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #013e7a;
    }
</style>
""", unsafe_allow_html=True)

# Input section
st.markdown("## 📝 Enter Match Details")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Home Team", teams)
    home_form = st.slider("📊 Home Team Form (0=poor, 5=excellent)", 0.0, 5.0, 2.5, step=0.1)
    home_rank = st.slider("🏅 Home Team Rank (1=best, 20=worst)", 1, 20, 10)

with col2:
    away_team = st.selectbox("✈️ Away Team", teams)
    away_form = st.slider("📊 Away Team Form (0=poor, 5=excellent)", 0.0, 5.0, 2.5, step=0.1)
    away_rank = st.slider("🏅 Away Team Rank (1=best, 20=worst)", 1, 20, 10)

# Predict button
if st.button("🏁 Predict Result"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'home_team_encoded': [le_team.transform([home_team])[0]],
        'away_team_encoded': [le_team.transform([away_team])[0]],
        'home_team_form': [home_form],
        'away_team_form': [away_form],
        'home_team_rank': [home_rank],
        'away_team_rank': [away_rank],
    })

    prediction = model.predict(input_df)
    result = le_result.inverse_transform(prediction)[0]

    # Show prediction
    st.success(f"🏆 **Predicted Result:** {result}")

# Footer
st.markdown("---")
st.caption("⚽ Made with football knowledge using Streamlit by Tshedza Tshikovhi")
