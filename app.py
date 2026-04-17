import streamlit as st
import joblib
from ai_engine import run_ai_engine

# UI Configuration
st.set_page_config(page_title="AI Symptom Assistant", page_icon="🩺", layout="centered")

st.title("🩺 AI Medical Symptom Assistant")
st.write("Select your symptoms below to receive safe, AI-powered medical guidance.")
st.info("Disclaimer: This AI assistant does not diagnose diseases. It only offers informational support. In case of emergency, please visit a hospital immediately.")

# Load Symptoms List for Dropdown
encoder_cols = joblib.load('encoder.pkl')
# Formatting symptoms to look clean (e.g., 'chest_pain' -> 'Chest Pain')
formatted_symptoms = [sym.replace('_', ' ').title() for sym in encoder_cols]

st.markdown("**💡 Tip:** Enter up to 4 to 5 symptoms for better and more accurate results.")
# Symptom Input Box
selected_symptoms_formatted = st.multiselect(
    "Select your symptoms (You can select multiple):",
    options=formatted_symptoms,
    help="Type to search for symptoms like 'Fever', 'Cough', etc."
)

# Convert formatted back to original code format for the ML model
selected_symptoms = [sym.replace(' ', '_').lower() for sym in selected_symptoms_formatted]

# Submit Button
if st.button("Analyze Symptoms 🔍"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom to analyze.")
    else:
        with st.spinner('Analyzing symptoms through ML & AI...'):
            try:
                # Backend Logic call 
                disease, logic_info, explanation = run_ai_engine(selected_symptoms)
                
                # Results Panel 
                st.success("Analysis Complete!")
                
                # Display Urgency and Doctor Type
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Condition", value=disease)
                with col2:
                    # Urgency color coding
                    urgency_color = "red" if logic_info['urgency'] == "Emergency" else ("orange" if logic_info['urgency'] == "Moderate" else "green")
                    st.markdown(f"**Urgency Level:** <span style='color:{urgency_color}'>{logic_info['urgency']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Recommended Doctor:** {logic_info['doctor']}")
                
                # Display LLM Explanation
                st.subheader("💡 AI Medical Advice")
                st.write(explanation)
                
            except Exception as e:
                st.error(f"Error when processing backend: {e}")
