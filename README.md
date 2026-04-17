# AI-Medical-Assistant

> ** Live Demo:** [Experience the App on Hugging Face Spaces](https://mrabees-smart-medical-assistant.hf.space)

## Project Overview
The **Smart Medical Symptom Assistant** is an AI-powered diagnostic tool. It is designed to analyze multiple user-inputted symptoms and predict potential medical conditions using a trained Machine Learning model. The system not only predicts diseases but also provides the urgency level, precautions, and the recommended physician type to guide the user.

## Key Features
* **Multi-Symptom Analysis:** Users can select up to 5 symptoms from a comprehensive medical database.
* **Accurate Predictions:** Powered by a Random Forest classifier trained on a robust clinical dataset.
* **LLM Integration:** Llama-3 for intelligent, context-aware medical advice generation.
* **Actionable Output:** Provides immediate precautions and urgency levels (e.g., Moderate, High).
* **Interactive UI:** A clean, responsive, and user-friendly interface built with Streamlit.

## Tech Stack
* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Model)
* **Generative AI:** Llama-3 (via API)
* **Data Processing:** Pandas, NumPy
* **Deployment:** Hugging Face Spaces

## Repository Structure
* `app.py`: Main Streamlit application file handling the UI.
* `ai_engine.py`: Backend logic, model loading, and API integration.
* `*.pkl files`: Pickled Random Forest model and label encoders for inference.
* `*.csv files`: Cleaned datasets, symptom descriptions, and precautions mapping.
* `requirements.txt`: Project dependencies.

##  How to Run Locally

1. **Clone the repository:**

2. **Install the dependencies:**
* pip install -r requirements.txt

3. **Run the Streamlit app**
* streamlit run app.py
