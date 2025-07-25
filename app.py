# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import warnings
# warnings.filterwarnings("ignore")

# # --- Load model and supporting files ---
# model = joblib.load('random_forest_model.pkl')
# model_features = joblib.load('model_features.pkl')
# categorical_options = joblib.load('categorical_options.pkl')
# digital_readiness_map = joblib.load('digital_readiness_mapping.pkl')

# st.title("Student Performance Predictor")
# st.write("Fill in the details below to predict a student's average total score, digital readiness, and at-risk status.")

# # --- Collect user input ---
# user_input = {}
# for col, options in categorical_options.items():
#     user_input[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}:", options)

# # --- Preprocessing function ---
# def preprocess_input(input_data):
#     input_df = pd.DataFrame([input_data])

#     # Calculate digital_readiness_score
#     input_df['digital_readiness_score'] = 0
#     for col, mapping in digital_readiness_map.items():
#         if col in input_df.columns:
#             norm_mapping = {str(k).strip().lower(): v for k, v in mapping.items()}
#             normalized_value = str(input_df[col].iloc[0]).strip().lower()
#             input_df['digital_readiness_score'] += norm_mapping.get(normalized_value, 0)

#     # One-hot encode categorical columns
#     dummy_df = pd.get_dummies(input_df, columns=categorical_options.keys(), drop_first=True)

#     # Align to model features
#     processed_df = pd.DataFrame(0, index=[0], columns=model_features)
#     for col in dummy_df.columns:
#         if col in processed_df.columns:
#             processed_df[col] = dummy_df[col]
#     return processed_df, int(input_df['digital_readiness_score'].iloc[0])

# # --- Prediction and Output ---
# if st.button("Predict Performance"):
#     with st.spinner("Predicting..."):
#         processed_input, digital_score = preprocess_input(user_input)
#         try:
#             prediction = model.predict(processed_input)[0]
#         except Exception as e:
#             st.error(f"Prediction failed: {e}")
#             st.stop()

#     # --- Score Prediction Section ---
#     st.header("🎯 Score Prediction")
#     st.success(f"Predicted Average Total Score: **{prediction:.2f}**")

#     # --- At Risk Students Section ---
#     # --- At Risk Students Section ---
# st.header("⚠️ At Risk Status")
# st.info("**At Risk:** Students predicted to score below the threshold are considered at risk and may need intervention.")

# # Sidebar slider for threshold
# at_risk_threshold = st.sidebar.slider(
#     "Set At-Risk Score Threshold", min_value=0, max_value=100, value=50, step=1
# )

# is_at_risk = prediction <= at_risk_threshold
# if is_at_risk:
#     st.error("This student is predicted to be AT RISK.")
# else:
#     st.info("This student is NOT predicted to be at risk.")

#     # --- Digital Readiness Section ---
#     st.header("💻 Digital Readiness")
#     st.info("**Digital Readiness:** A higher score means better access to digital resources and technology.")
#     st.info(f"Digital Readiness Score: **{digital_score}**")
#     if digital_score >= 7:
#         st.success("Excellent digital readiness!")
#     elif digital_score >= 4:
#         st.info("Moderate digital readiness.")
#     else:
#         st.warning("Low digital readiness. Consider improving access to digital resources.")

#     # --- User Input Summary ---
#     st.header("📝 Your Input Summary")
#     st.json(user_input)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- Load model and supporting files ---
model = joblib.load('random_forest_model.pkl')
model_features = joblib.load('model_features.pkl')
categorical_options = joblib.load('categorical_options.pkl')
digital_readiness_map = joblib.load('digital_readiness_mapping.pkl')

st.title("Student Performance Predictor")
st.write("Fill in the details below to predict a student's average total score, digital readiness, and at-risk status.")

# --- Collect user input ---
user_input = {}
for col, options in categorical_options.items():
    user_input[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}:", options)

# --- Preprocessing function ---
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])

    # Calculate digital_readiness_score
    input_df['digital_readiness_score'] = 0
    for col, mapping in digital_readiness_map.items():
        if col in input_df.columns:
            norm_mapping = {str(k).strip().lower(): v for k, v in mapping.items()}
            normalized_value = str(input_df[col].iloc[0]).strip().lower()
            input_df['digital_readiness_score'] += norm_mapping.get(normalized_value, 0)

    # One-hot encode categorical columns
    dummy_df = pd.get_dummies(input_df, columns=categorical_options.keys(), drop_first=True)

    # Align to model features
    processed_df = pd.DataFrame(0, index=[0], columns=model_features)
    for col in dummy_df.columns:
        if col in processed_df.columns:
            processed_df[col] = dummy_df[col]
    return processed_df, int(input_df['digital_readiness_score'].iloc[0])

# --- Prediction and Output ---
if st.button("Predict Performance"):
    with st.spinner("Predicting..."):
        processed_input, digital_score = preprocess_input(user_input)
        try:
            prediction = model.predict(processed_input)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # --- Score Prediction Section ---
    st.header("🎯 Score Prediction")
    st.success(f"Predicted Average Total Score: **{prediction:.2f}**")

    # --- At Risk Students Section ---
    st.header("⚠️ At Risk Status")
    st.info("**At Risk:** Students predicted to score below the threshold are considered at risk and may need intervention.")

    at_risk_threshold = st.sidebar.slider(
        "Set At-Risk Score Threshold", min_value=0, max_value=100, value=50, step=1
    )

    is_at_risk = prediction <= at_risk_threshold
    if is_at_risk:
        st.error("This student is predicted to be AT RISK.")
    else:
        st.info("This student is NOT predicted to be at risk.")

    # --- Digital Readiness Section ---
    st.header("💻 Digital Readiness")
    st.info("**Digital Readiness:** A higher score means better access to digital resources and technology.")
    st.info(f"Digital Readiness Score: **{digital_score}**")
    if digital_score >= 7:
        st.success("Excellent digital readiness!")
    elif digital_score >= 4:
        st.info("Moderate digital readiness.")
    else:
        st.warning("Low digital readiness. Consider improving access to digital resources.")

    # --- User Input Summary ---
    st.header("📝 Your Input Summary")
    st.json(user_input)