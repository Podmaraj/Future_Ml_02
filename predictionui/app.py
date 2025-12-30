import streamlit as st
import pandas as pd
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Spotify Churn Prediction",
    page_icon="🎧",
    layout="wide"
)

# LOAD CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================================
# LOAD MODEL & DATA
# ================================
model = joblib.load("spotify_churn_model.pkl")
scaler = joblib.load("spotify_scaler.pkl")
df = pd.read_csv("spotify_data.csv")
df.columns = df.columns.str.strip()

FEATURES = [
    "Age", "Gender", "spotify_usage_period",
    "spotify_listening_device", "spotify_subscription_plan",
    "premium_sub_willingness", "preferred_listening_content",
    "fav_music_genre", "music_time_slot",
    "music_lis_frequency", "music_recc_rating"
]

# Prepare dummies from training data
X = pd.get_dummies(df[FEATURES], drop_first=True)
MODEL_COLUMNS = X.columns

# ================================
# HEADER UI

st.markdown("""
<div class="header">
    <h1>🎧 Spotify Churn Prediction</h1>
    <p>Predict customer churn using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("🎛 User Profile")

user_input = {
    "Age": st.sidebar.slider("Age", 10, 80, 25),
    "Gender": st.sidebar.selectbox("Gender", df["Gender"].unique()),
    "spotify_usage_period": st.sidebar.selectbox("Usage Period", df["spotify_usage_period"].unique()),
    "spotify_listening_device": st.sidebar.selectbox("Listening Device", df["spotify_listening_device"].unique()),
    "spotify_subscription_plan": st.sidebar.selectbox("Subscription Plan", df["spotify_subscription_plan"].unique()),
    "premium_sub_willingness": st.sidebar.selectbox("Premium Willingness", ["Yes", "No"]),
    "preferred_listening_content": st.sidebar.selectbox("Content Type", df["preferred_listening_content"].unique()),
    "fav_music_genre": st.sidebar.selectbox("Music Genre", df["fav_music_genre"].unique()),
    "music_time_slot": st.sidebar.selectbox("Time Slot", df["music_time_slot"].unique()),
    "music_lis_frequency": st.sidebar.selectbox("Listening Frequency", df["music_lis_frequency"].unique()),
    "music_recc_rating": st.sidebar.slider("Recommendation Rating", 1, 5, 3)
}


# PREDICTION LOGIC
if st.sidebar.button("🚀 Predict Churn"):

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical variables
    input_dummies = pd.get_dummies(input_df)

    # Add missing columns as 0
    for col in MODEL_COLUMNS:
        if col not in input_dummies.columns:
            input_dummies[col] = 0

    # Reorder columns to match model
    input_dummies = input_dummies[MODEL_COLUMNS]

    # Scale the input
    input_scaled = scaler.transform(input_dummies)

    # Predict churn probability
    churn_prob = model.predict_proba(input_scaled)[0][1]

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # RESULT UI LOGIC
    if churn_prob < 0.3:
        status, glow_class, icon = "Low Risk", "glow-low", "✅"
        retention_list = [
            "Offer regular playlists",
            "Light push notifications",
            "Basic engagement campaigns"
        ]
    elif churn_prob < 0.7:
        status, glow_class, icon = "Medium Risk", "glow-medium", "⚠️"
        retention_list = [
            "Premium trial discounts",
            "Enhanced recommendations",
            "Targeted push notifications"
        ]
    else:
        status, glow_class, icon = "High Risk", "glow-high", "🚨"
        retention_list = [
            "Exclusive premium offers",
            "Personalized playlists",
            "High-priority notifications",
            "Special engagement campaigns"
        ]

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="{glow_class}">
            <h2>{icon} {status}</h2>
            <h3>Churn Probability: {churn_prob:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="retention-card">
            <h3>📌 Retention Strategy</h3>
            <ul>
                {"".join([f"<li>{item}</li>" for item in retention_list])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<div class="footer">
Built with ❤️ using Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
