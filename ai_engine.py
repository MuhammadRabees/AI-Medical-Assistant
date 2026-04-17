import joblib
import pandas as pd
from groq import Groq

# Load Models
model = joblib.load('model.pkl')
encoder_cols = joblib.load('encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load CSVs
description_df = pd.read_csv('symptom_Description.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')

# Groq Setup (Enter your API Key here)
client = Groq(api_key="gsk_ ----")

def get_disease_logic(disease_name):
    disease_name = disease_name.strip()
    desc = description_df[description_df['Disease'] == disease_name]['Description'].values
    full_desc = desc[0] if len(desc) > 0 else "No specific description available."
    
    prec = precaution_df[precaution_df['Disease'] == disease_name].iloc[:, 1:].values.tolist()
    flat_prec = [str(item) for sublist in prec for item in sublist if str(item) != 'nan']

    emergency_list = ['Dengue', 'Heart attack', 'Stroke', 'Pneumonia', 'Hypertension']
    moderate_list = ['Malaria', 'Typhoid', 'Jaundice', 'Tuberculosis', 'Asthma', 'Chicken pox']
    
    if disease_name in emergency_list:
        urgency = "Emergency"
        doctor = "Emergency Specialist / ER"
    elif disease_name in moderate_list:
        urgency = "Moderate"
        doctor = "General Physician"
    else:
        urgency = "Mild"
        doctor = "General Physician / Specialist"
        
    return {'urgency': urgency, 'doctor': doctor, 'description': full_desc, 'precautions': flat_prec}

def get_ai_response(disease_name, user_symptoms, logic_info):
    prompt = f"""
    User is experiencing these symptoms: {', '.join(user_symptoms)}.
    The ML model predicted the possible condition is: {disease_name}.
    Medical Description: {logic_info['description']}
    Standard Precautions: {', '.join(logic_info['precautions'])}
    Urgency Level: {logic_info['urgency']}
    Recommended Doctor: {logic_info['doctor']}
    
    Act as a helpful and safe AI Medical Symptom Assistant. 
    1. Write a short, clear, and empathetic response explaining the condition.
    2. Provide the best safe precautions from the list above.
    3. Clearly state the urgency level and the type of doctor they should see.
    4. Strict Disclaimer: End the response by stating clearly that you are an AI, not a doctor, and this is not a final diagnosis.
    5. IMPORTANT: If the predicted condition is severe (like Tuberculosis, Dengue, Malaria, Heart Attack, Typhoid, Chicken Pox) but the user only provided 1 to 3 very common symptoms (like fever, cough, headache), mention that it could likely be a common viral infection or seasonal bug, and the severe prediction is just a worst-case possibility based on limited data. Do not alarm the user.
    """
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return completion.choices[0].message.content

def run_ai_engine(input_symptoms):
    input_vector = [1 if sym in input_symptoms else 0 for sym in encoder_cols]
    prediction_idx = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([prediction_idx])[0]
    logic_info = get_disease_logic(predicted_disease)
    final_advice = get_ai_response(predicted_disease, input_symptoms, logic_info)
    return predicted_disease, logic_info, final_advice
