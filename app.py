import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS to allow frontend to connect
# --- Configuration for loading models ---
MODEL_DIR = 'trained_models'
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
DISTRICT_ENCODING_PATH = os.path.join(MODEL_DIR, 'district_mean_encoding.pkl')

# Path to the original dataset used for feature extraction (Crucial for getting other input features)
ORIGINAL_DF_PATH = 'FINALONE___.xlsx' # Assuming this is in the same directory
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
# Global variables to hold the loaded models and data
ensemble_model = None
scaler = None
label_encoders = {}
district_mean_encoding = {}
original_df_for_features = None # Will store a version of FINALONE___.xlsx for feature extraction
feature_columns_global = None # Store feature columns after initial load
def load_models_and_data():
    global ensemble_model, scaler, label_encoders, district_mean_encoding, original_df_for_features, feature_columns_global
    
    print("Loading models and data for the Flask app...")
    try:
        ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        district_mean_encoding = joblib.load(DISTRICT_ENCODING_PATH)
        # Load the original_df (FINALONE___.xlsx) and preprocess it for feature extraction
        # This needs to be consistent with how it was processed during training for prediction
        temp_df = pd.read_excel(ORIGINAL_DF_PATH)
        temp_df.drop(columns=['temperature_2m_mean', 'rain_sum', 'soil_moisture_7_to_28cm', 'soil_temperature_7_to_28cm'], inplace=True)
        
        # Apply the same label encoding for 'season', 'Crop', 'State_Name' to original_df_for_features
        # Make sure 'season' is stripped before encoding if that was done during training
        temp_df['season'] = label_encoders['season'].transform(temp_df['season'].str.strip())
        # For Crop and State_Name, they might not be directly used for input features but
        # if any part of your feature extraction logic depends on them being encoded, apply it.
        # For now, let's assume 'season' encoding is the primary one for filtering.
        temp_df['Crop'] = label_encoders['Crop'].transform(temp_df['Crop'])
        temp_df['State_Name'] = label_encoders['State_Name'].transform(temp_df['State_Name'])
        original_df_for_features = temp_df.copy() # Store this for feature extraction
        
        # Define feature columns here, consistent with how the model was trained
        # Ensure 'district_encoded' is available in original_df_for_features if needed later
        # (though it's usually calculated from 'df' which is not original_df_for_features)
        # For simplicity, we assume original_df_for_features has all needed columns before scaling
        feature_columns_global = ['season', 'Production', 'relative_humidity_2m', 'temperature_2m_min', 
                                  'soil_moisture_0_to_7cm', 'Area', 'rain', 'temperature_2m', 
                                  'nitrogen_share', 'temperature_2m_max', 'district_encoded']
        print("Models and data loaded successfully.")
    except FileNotFoundError:
        print("Error: Model files or original data file not found. Please ensure 'train_and_predict.py' has been run successfully and files are in the correct place.")
        exit() # Exit if models can't be loaded, server cannot run without them
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        exit()
# Call this function once when the app starts
with app.app_context(): # Ensure app context for potential app.config usage
    load_models_and_data()
# --- Prediction functions (copied from your Python script, now using global loaded objects) ---
def predict_top_n_crops(month, season_str, district_str, df_reference, label_encoders_obj, scaler_obj, model_obj, feature_cols, district_enc_map, n=3):
    district_lower = district_str.strip().lower()
    season_stripped = season_str.strip()
    df_ref_filtered = df_reference[df_reference['district'].str.strip().str.lower() == district_lower].copy().head(1)
    if df_ref_filtered.empty:
        # Fallback if specific district not found in original_df for features
        # Or perhaps fetch average values for that district if not found specific
        raise ValueError(f"District '{district_str}' not found in dataset for feature extraction.")
    
    if season_stripped not in label_encoders_obj['season'].classes_:
        raise ValueError(f"Season '{season_stripped}' not recognized. Available: {list(label_encoders_obj['season'].classes_)}")
    encoded_season = label_encoders_obj['season'].transform([season_stripped])[0]
    df_ref_filtered['season'] = encoded_season
    # df_ref_filtered['month'] = month # Placeholder, not directly used in X for traditional prediction
    original_district_name_for_encoding = df_ref_filtered['district'].values[0]
    if original_district_name_for_encoding not in district_enc_map:
        # Fallback for district encoding if not found
        df_ref_filtered['district_encoded'] = np.mean(list(district_enc_map.values()))
    else:
        df_ref_filtered['district_encoded'] = district_enc_map[original_district_name_for_encoding]
        
    input_df_features = df_ref_filtered[feature_cols]
    
    # Ensure input_df_features has all required columns before scaling
    # If any column is missing, it will cause an error during scaling.
    # This check helps debug:
    missing_cols = [col for col in feature_cols if col not in input_df_features.columns]
    if missing_cols:
        raise ValueError(f"Missing required features for prediction: {missing_cols}")
    input_scaled = scaler_obj.transform(input_df_features)
    
    probabilities = model_obj.predict_proba(input_scaled)[0]
    top_n_indices = np.argsort(probabilities)[::-1][:n]
    
    top_n_crops = []
    for idx in top_n_indices:
        crop_name = label_encoders_obj['Crop'].inverse_transform([idx])[0]
        probability = probabilities[idx]
        top_n_crops.append({"name": crop_name, "probability": probability})
    
    return top_n_crops
def suggest_top_n_exotic_crops(features, exotic_df_ref, n=2):
    crop_scores = []
    for _, row in exotic_df_ref.iterrows():
        try:
            crop_name = row['Crop']
            temp_min = float(row['Temperature ©'])
            temp_max = float(row['Unnamed: 15'])
            ph = float(row['pH level'])
            humidity_min = float(row['Humidity (%)'])
            humidity_max = float(row['Unnamed: 17'])
            score = 0
            if not (temp_min <= features['temperature_2m_min'] <= temp_max):
                score += min(abs(features['temperature_2m_min'] - temp_min), abs(features['temperature_2m_min'] - temp_max))
            if not (ph - 0.5 <= features['ph'] <= ph + 0.5):
                score += abs(features['ph'] - ph)
            if not (humidity_min <= features['relative_humidity_2m'] <= humidity_max):
                score += min(abs(features['relative_humidity_2m'] - humidity_min), abs(features['relative_humidity_2m'] - humidity_max))
            
            crop_scores.append((crop_name, score))
        except (ValueError, TypeError): # Handle cases where conversion to float fails
            continue
    
    crop_scores.sort(key=lambda x: x[1])
    top_n_exotic_crops = [crop for crop, score in crop_scores[:n]]
    return top_n_exotic_crops if top_n_exotic_crops else ["No suitable exotic crop found"]
# Load exotic_df outside the function as it's static
exotic_df_global = pd.read_excel("Growloc_Plant details_Updated_May 2023.xlsx")
@app.route('/predict_crops', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400
    month = data.get('month')
    season = data.get('season')
    district = data.get('district')
    if not all([month, season, district]):
        return jsonify({"error": "Missing month, season, or district"}), 400
    try:
        # Predict traditional crops
        top_3_traditional_crops = predict_top_n_crops(
            month=int(month), # Ensure month is int
            season_str=season,
            district_str=district,
            df_reference=original_df_for_features,
            label_encoders_obj=label_encoders,
            scaler_obj=scaler,
            model_obj=ensemble_model,
            feature_cols=feature_columns_global,
            district_enc_map=district_mean_encoding,
            n=3
        )
        # Extract feature values for exotic suggestion
        # Filter original_df_for_features using the original string value of district
        # and the encoded season value (which is already in original_df_for_features)
        feature_row = original_df_for_features[
            (original_df_for_features['district'].str.strip().str.lower() == district.strip().lower()) &
            (original_df_for_features['season'] == label_encoders['season'].transform([season.strip()])[0])
        ].head(1)
        top_2_exotic_crops = ["No suitable exotic crop found (input row missing for features)"]
        if not feature_row.empty:
            exotic_input_features = {
                'temperature_2m_min': feature_row['temperature_2m_min'].values[0],
                'relative_humidity_2m': feature_row['relative_humidity_2m'].values[0],
                'ph': 6.5 # assumed pH since it's not available in traditional dataset
            }
            top_2_exotic_crops = suggest_top_n_exotic_crops(exotic_input_features, exotic_df_global, n=2)
        return jsonify({
            "traditional_crops": top_3_traditional_crops,
            "exotic_crops": top_2_exotic_crops
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "An internal server error occurred during prediction."}), 500
if __name__ == '__main__':
    # To run in production, you'd use a WSGI server like Gunicorn or uWSGI
    # For development, run directly:
    app.run(debug=True, port=5000) # port 5000 is common for Flask dev server
