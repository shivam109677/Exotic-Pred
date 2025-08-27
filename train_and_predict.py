# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier
# import joblib # Import joblib for saving/loading models
# import os # Import os for checking file existence

# # --- Configuration for saving/loading models ---
# MODEL_DIR = 'trained_models'
# ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
# SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
# LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
# DISTRICT_ENCODING_PATH = os.path.join(MODEL_DIR, 'district_mean_encoding.pkl')
# PH_ENCODING_PATH = os.path.join(MODEL_DIR, 'ph_mean_encoding.pkl') # New path for pH encoding

# # Ensure the model directory exists
# os.makedirs(MODEL_DIR, exist_ok=True)

# # --- Load datasets ---
# df = pd.read_excel("FINALONE___.xlsx")
# exotic_df = pd.read_excel("Growloc_Plant details_Updated_May 2023.xlsx")

# # Drop unused columns from traditional dataset
# df.drop(columns=['temperature_2m_mean', 'rain_sum', 'soil_moisture_7_to_28cm', 'soil_temperature_7_to_28cm'], inplace=True)

# # Backup original for reference (needed for predict_top_n_crops)
# original_df = df.copy()


# X = df[['season', 'Production', 'relative_humidity_2m', 'temperature_2m_min', 'soil_moisture_0_to_7cm',
#              'Area', 'rain', 'temperature_2m', 'nitrogen_share', 'temperature_2m_max', 'district_encoded']]
# y = df['Crop']

# # --- Check if models exist, if not, train them ---
# if (os.path.exists(ENSEMBLE_MODEL_PATH) and
#     os.path.exists(SCALER_PATH) and
#     os.path.exists(LABEL_ENCODERS_PATH) and
#     os.path.exists(DISTRICT_ENCODING_PATH) and
#     os.path.exists(PH_ENCODING_PATH)): # Check for pH encoding

#     print("Loading pre-trained models and components...")
#     ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     label_encoders = joblib.load(LABEL_ENCODERS_PATH)
#     district_mean_encoding = joblib.load(DISTRICT_ENCODING_PATH)
#     ph_mean_encoding = joblib.load(PH_ENCODING_PATH) # Load pH encoding

#     # Re-encode original_df for consistency with loaded label_encoders
#     original_df['season'] = label_encoders['season'].transform(original_df['season'].str.strip())
#     original_df['Crop'] = label_encoders['Crop'].transform(original_df['Crop'])
#     original_df['State_Name'] = label_encoders['State_Name'].transform(original_df['State_Name'])



# else:
#     print("Pre-trained models not found. Training models...")
#     # Clean and encode categorical columns
#     original_df['season'] = original_df['season'].str.strip()

#     label_encoders = {}

#     le_season = LabelEncoder()
#     original_df['season'] = le_season.fit_transform(original_df['season'])
#     label_encoders['season'] = le_season
#     df['season'] = original_df['season'] # Ensure df is updated for district encoding

#     le_crop = LabelEncoder()
#     original_df['Crop'] = le_crop.fit_transform(original_df['Crop'])
#     label_encoders['Crop'] = le_crop
#     df['Crop'] = original_df['Crop'] # Ensure df is updated for district encoding

#     le_state = LabelEncoder()
#     original_df['State_Name'] = le_state.fit_transform(original_df['State_Name'])
#     label_encoders['State_Name'] = le_state
#     df['State_Name'] = original_df['State_Name'] # Ensure df is updated for district encoding

#     # District mean encoding
#     df['district_encoded'] = df.groupby('district')['Crop'].transform('mean')
#     df['district_clean'] = df['district'].str.strip().str.lower()

#     # Create district encoding map (from the 'df' that has the 'Crop' column encoded)
#     district_mean_encoding = df.groupby('district')['Crop'].mean().to_dict()

#     # --- New: Calculate and store mean pH for each district/season ---
#     # Assuming 'ph' column is available or can be derived.
#     # For demonstration, let's create a dummy 'ph' column in df
#     # In a real scenario, you would load or generate actual pH data.
#     if 'ph' not in df.columns:
#         # Generate random pH values between 5.5 and 7.5 for demonstration
#         df['ph'] = np.random.uniform(5.5, 7.5, size=len(df))
    
#     # Calculate mean pH for each unique combination of district and season
#     ph_mean_encoding = df.groupby(['district', 'season'])['ph'].mean().to_dict()


#     # Prepare features and scale
#     X = df[['season', 'Production', 'relative_humidity_2m', 'temperature_2m_min', 'soil_moisture_0_to_7cm',
#              'Area', 'rain', 'temperature_2m', 'nitrogen_share', 'temperature_2m_max', 'district_encoded']]
#     y = df['Crop']

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # Train models
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
#     gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
#     dt = DecisionTreeClassifier(random_state=42)
#     catboost = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, random_state=42, verbose=0)

#     rf.fit(X_train, y_train)
#     xgb.fit(X_train, y_train)
#     gb.fit(X_train, y_train)
#     dt.fit(X_train, y_train)
#     catboost.fit(X_train, y_train)

#     # Ensemble model
#     ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('dt', dt), ('catboost', catboost)], voting='soft')
#     ensemble.fit(X_train, y_train)

#     # --- Save trained models and components ---
#     print("Saving trained models and components...")
#     joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)
#     joblib.dump(scaler, SCALER_PATH)
#     joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
#     joblib.dump(district_mean_encoding, DISTRICT_ENCODING_PATH)
#     joblib.dump(ph_mean_encoding, PH_ENCODING_PATH) # Save pH encoding
#     print("Models and components saved successfully.")

# # Feature columns for training (must be consistent with how the model was trained)
# feature_columns = X.columns.tolist()

# # --- Prediction functions (remain the same) ---
# def predict_top_n_crops(month, season, district, df_reference, label_encoders, scaler, model, feature_columns, district_encoding_map, n=3):
#     district = district.strip().lower()
#     season = season.strip()

#     df_reference_filtered = df_reference[df_reference['district'].str.strip().str.lower() == district].copy().head(1)

#     if df_reference_filtered.empty:
#         raise ValueError(f"District '{district}' not found in dataset for feature extraction.")

#     if season not in label_encoders['season'].classes_:
#         raise ValueError(f"Season '{season}' not recognized. Available: {list(label_encoders['season'].classes_)}")

#     encoded_season = label_encoders['season'].transform([season])[0]
#     df_reference_filtered['season'] = encoded_season
#     df_reference_filtered['month'] = month # Placeholder, not directly used in X

#     original_district_name_for_encoding = df_reference_filtered['district'].values[0]
#     if original_district_name_for_encoding not in district_encoding_map:
#         df_reference_filtered['district_encoded'] = np.mean(list(district_encoding_map.values()))
#     else:
#         df_reference_filtered['district_encoded'] = district_encoding_map[original_district_name_for_encoding]
        
#     input_df_features = df_reference_filtered[feature_columns]

#     input_scaled = scaler.transform(input_df_features)
    
#     probabilities = model.predict_proba(input_scaled)[0]
#     top_n_indices = np.argsort(probabilities)[::-1][:n]
    
#     top_n_crops = [(label_encoders['Crop'].inverse_transform([idx])[0], probabilities[idx]) for idx in top_n_indices]
    
#     return top_n_crops

# def suggest_top_n_exotic_crops(features, exotic_df, n=2):
#     crop_scores = []

#     for _, row in exotic_df.iterrows():
#         try:
#             crop_name = row['Crop']
#             temp_min = float(row['Temperature ©'])
#             temp_max = float(row['Unnamed: 15'])
#             ph = float(row['pH level'])
#             humidity_min = float(row['Humidity (%)'])
#             humidity_max = float(row['Unnamed: 17'])

#             score = 0
#             # Temperature score
#             if not (temp_min <= features['temperature_2m_min'] <= temp_max):
#                 score += min(abs(features['temperature_2m_min'] - temp_min), abs(features['temperature_2m_min'] - temp_max))
            
#             # pH score
#             # A tighter range around optimal pH, more sensitive to deviations
#             if not (ph - 0.25 <= features['ph'] <= ph + 0.25): # Smaller tolerance for pH
#                 score += abs(features['ph'] - ph) * 2 # Increase penalty for pH deviation

#             # Humidity score
#             if not (humidity_min <= features['relative_humidity_2m'] <= humidity_max):
#                 score += min(abs(features['relative_humidity_2m'] - humidity_min), abs(features['relative_humidity_2m'] - humidity_max))
            
#             # Introduce a small random perturbation to scores to break ties
#             score += np.random.uniform(0, 0.001) # Small random noise to break ties

#             crop_scores.append((crop_name, score))
#         except Exception as e:
#             # print(f"Skipping exotic crop due to error: {e} for row: {row.get('Crop', 'Unknown')}")
#             continue
    
#     crop_scores.sort(key=lambda x: x[1])
#     top_n_exotic_crops = [crop for crop, score in crop_scores[:n]]

#     return top_n_exotic_crops if top_n_exotic_crops else ["No suitable exotic crop found"]

# # --- User inputs (example) ---
# user_month = 7
# user_season = 'Kharif'
# user_district = 'Akola' 

# # --- Make predictions ---
# print(f"\nPredicting for {user_district}, {user_season} season, Month {user_month}:")

# # Predict top 3 traditional crops
# top_3_traditional_crops = predict_top_n_crops(
#     month=user_month,
#     season=user_season,
#     district=user_district,
#     df_reference=original_df, # Use original_df for features
#     label_encoders=label_encoders,
#     scaler=scaler,
#     model=ensemble,
#     feature_columns=feature_columns,
#     district_encoding_map=district_mean_encoding,
#     n=3
# )

# # Extract feature values for exotic suggestion
# # Ensure 'district' and 'season' columns are in original_df correctly before filtering
# # This filter is based on the original string values of district and the *encoded* season value
# feature_row = original_df[
#     (original_df['district'].str.strip().str.lower() == user_district.strip().lower()) &
#     (original_df['season'] == label_encoders['season'].transform([user_season.strip()])[0])
# ].head(1)

# if not feature_row.empty:
#     # --- New: Get pH from the mean encoding ---
#     # Ensure the key for ph_mean_encoding is a tuple of (district_name, encoded_season)
#     district_name_for_ph = feature_row['district'].values[0]
#     encoded_season_for_ph = feature_row['season'].values[0] # This is already encoded
    
#     # Check if the exact (district, season) combination exists in ph_mean_encoding
#     if (district_name_for_ph, encoded_season_for_ph) in ph_mean_encoding:
#         calculated_ph = ph_mean_encoding[(district_name_for_ph, encoded_season_for_ph)]
#     else:
#         # Fallback if specific district-season pH not found
#         # You might want a more sophisticated fallback (e.g., district average, global average)
#         print(f"Warning: pH for district '{user_district}' and season '{user_season}' not found. Using global average pH.")
#         calculated_ph = np.mean(list(ph_mean_encoding.values())) # Fallback to global average pH

#     exotic_input_features = {
#         'temperature_2m_min': feature_row['temperature_2m_min'].values[0],
#         'relative_humidity_2m': feature_row['relative_humidity_2m'].values[0],
#         'ph': calculated_ph  # Use the calculated pH
#     }

#     top_2_exotic_crops = suggest_top_n_exotic_crops(exotic_input_features, exotic_df, n=2)
# else:
#     top_2_exotic_crops = ["No suitable exotic crop found (input row missing for features)"]

# print("\n--- Prediction Results ---")
# print("Top 3 Traditional Crops and their probabilities:")
# for crop, prob in top_3_traditional_crops:
#     print(f"- {crop}")

# print("\nTop 2 Exotic Crops:")
# for crop in top_2_exotic_crops:
#     print(f"- {crop}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import joblib # Import joblib for saving/loading models
import os # Import os for checking file existence

# --- Configuration for saving/loading models ---
MODEL_DIR = 'trained_models'
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
DISTRICT_ENCODING_PATH = os.path.join(MODEL_DIR, 'district_mean_encoding.pkl')
PH_ENCODING_PATH = os.path.join(MODEL_DIR, 'ph_mean_encoding.pkl') # New path for pH encoding

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load datasets ---
df = pd.read_excel("FINALONE___.xlsx")
exotic_df = pd.read_excel("Growloc_Plant details_Updated_May 2023.xlsx")

# Drop unused columns from traditional dataset
df.drop(columns=['temperature_2m_mean', 'rain_sum', 'soil_moisture_7_to_28cm', 'soil_temperature_7_to_28cm'], inplace=True)

# Backup original for reference (needed for predict_top_n_crops)
original_df = df.copy()

# Define feature columns BEFORE the if/else block
# These are the columns expected by the trained model
feature_columns = ['season', 'Production', 'relative_humidity_2m', 'temperature_2m_min', 'soil_moisture_0_to_7cm',
                   'Area', 'rain', 'temperature_2m', 'nitrogen_share', 'temperature_2m_max', 'district_encoded']

# --- Check if models exist, if not, train them ---
if (os.path.exists(ENSEMBLE_MODEL_PATH) and
    os.path.exists(SCALER_PATH) and
    os.path.exists(LABEL_ENCODERS_PATH) and
    os.path.exists(DISTRICT_ENCODING_PATH) and
    os.path.exists(PH_ENCODING_PATH)): # Check for pH encoding

    print("Loading pre-trained models and components...")
    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    district_mean_encoding = joblib.load(DISTRICT_ENCODING_PATH)
    ph_mean_encoding = joblib.load(PH_ENCODING_PATH) # Load pH encoding

    # Re-encode original_df for consistency with loaded label_encoders
    # Only re-encode if the columns exist and are not already encoded
    if 'season' in original_df.columns and original_df['season'].dtype == object: # Check if it's still object type (not encoded)
        original_df['season'] = label_encoders['season'].transform(original_df['season'].str.strip())
    if 'Crop' in original_df.columns and original_df['Crop'].dtype == object:
        original_df['Crop'] = label_encoders['Crop'].transform(original_df['Crop'])
    if 'State_Name' in original_df.columns and original_df['State_Name'].dtype == object:
        original_df['State_Name'] = label_encoders['State_Name'].transform(original_df['State_Name'])
    
    # Recreate 'district_encoded' if not present in loaded original_df
    # This assumes 'district' column is still present in original_df
    if 'district_encoded' not in original_df.columns and 'district' in original_df.columns:
        original_df['district_encoded'] = original_df['district'].str.strip().str.lower().map(district_mean_encoding)
        # Handle potential NaNs if a district isn't in the encoding map
        original_df['district_encoded'].fillna(np.mean(list(district_mean_encoding.values())), inplace=True)


else:
    print("Pre-trained models not found. Training models...")
    # Clean and encode categorical columns
    original_df['season'] = original_df['season'].str.strip()

    label_encoders = {}

    le_season = LabelEncoder()
    original_df['season'] = le_season.fit_transform(original_df['season'])
    label_encoders['season'] = le_season
    df['season'] = original_df['season'] # Ensure df is updated for district encoding

    le_crop = LabelEncoder()
    original_df['Crop'] = le_crop.fit_transform(original_df['Crop'])
    label_encoders['Crop'] = le_crop
    df['Crop'] = original_df['Crop'] # Ensure df is updated for district encoding

    le_state = LabelEncoder()
    original_df['State_Name'] = le_state.fit_transform(original_df['State_Name'])
    label_encoders['State_Name'] = le_state
    df['State_Name'] = original_df['State_Name'] # Ensure df is updated for district encoding

    # District mean encoding
    # Ensure 'district' is cleaned before grouping
    df['district_clean'] = df['district'].str.strip().str.lower()
    df['district_encoded'] = df.groupby('district_clean')['Crop'].transform('mean')

    # Create district encoding map (from the 'df' that has the 'Crop' column encoded)
    district_mean_encoding = df.groupby('district_clean')['Crop'].mean().to_dict()
    # Apply to original_df as well for consistency
    original_df['district_encoded'] = original_df['district'].str.strip().str.lower().map(district_mean_encoding)
    original_df['district_encoded'].fillna(np.mean(list(district_mean_encoding.values())), inplace=True) # Handle new districts

    # --- New: Calculate and store mean pH for each district/season ---
    if 'ph' not in df.columns:
        # Generate random pH values between 5.5 and 7.5 for demonstration
        df['ph'] = np.random.uniform(5.5, 7.5, size=len(df))
    
    # Calculate mean pH for each unique combination of district and season
    ph_mean_encoding = df.groupby(['district', 'season'])['ph'].mean().to_dict()

    # Prepare features and scale
    X = df[feature_columns] # Use the globally defined feature_columns
    y = df['Crop']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    catboost = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, random_state=42, verbose=0)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    catboost.fit(X_train, y_train)

    # Ensemble model
    ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('dt', dt), ('catboost', catboost)], voting='soft')
    ensemble.fit(X_train, y_train)

    # --- Save trained models and components ---
    print("Saving trained models and components...")
    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    joblib.dump(district_mean_encoding, DISTRICT_ENCODING_PATH)
    joblib.dump(ph_mean_encoding, PH_ENCODING_PATH) # Save pH encoding
    print("Models and components saved successfully.")

# Feature columns for training (must be consistent with how the model was trained)
# This line is now redundant as feature_columns is defined at the beginning.
# feature_columns = X.columns.tolist() # Remove or comment out this line

# --- Prediction functions (remain the same) ---
def predict_top_n_crops(month, season, district, df_reference, label_encoders, scaler, model, feature_columns, district_encoding_map, n=3):
    district = district.strip().lower()
    season = season.strip()

    # Filter df_reference to get a row for the specified district and season
    # Use the string value of district and the *encoded* value for season
    # This means original_df needs to have its season column encoded if models are loaded
    
    # First, try to get a row that matches the exact district (case-insensitive)
    # and has the season already encoded (if models are loaded) or about to be encoded (if models are trained)
    df_reference_filtered = df_reference[df_reference['district'].str.strip().str.lower() == district].copy()
    
    # If the season column in df_reference_filtered is still object type, encode it for filtering
    if not df_reference_filtered.empty and df_reference_filtered['season'].dtype == object:
        # Check if the season is in the known classes of the label encoder
        if season not in label_encoders['season'].classes_:
            raise ValueError(f"Season '{season}' not recognized. Available: {list(label_encoders['season'].classes_)}")
        encoded_season_val = label_encoders['season'].transform([season])[0]
        df_reference_filtered['season'] = df_reference_filtered['season'].str.strip().apply(lambda x: label_encoders['season'].transform([x])[0])
        df_reference_filtered = df_reference_filtered[df_reference_filtered['season'] == encoded_season_val].head(1)
    elif not df_reference_filtered.empty and df_reference_filtered['season'].dtype != object: # Already encoded
        if season not in label_encoders['season'].classes_:
            raise ValueError(f"Season '{season}' not recognized. Available: {list(label_encoders['season'].classes_)}")
        encoded_season_val = label_encoders['season'].transform([season])[0]
        df_reference_filtered = df_reference_filtered[df_reference_filtered['season'] == encoded_season_val].head(1)
    else: # If district not found initially
        df_reference_filtered = pd.DataFrame() # Ensure it's empty if no district match

    if df_reference_filtered.empty:
        # Fallback: if no exact district-season match, try to find just the district
        df_reference_filtered = df_reference[df_reference['district'].str.strip().str.lower() == district].copy().head(1)
        if df_reference_filtered.empty:
            raise ValueError(f"District '{district}' not found in dataset for feature extraction.")
        
        # If we got a district row but not a specific season, we'll manually set the season
        if season not in label_encoders['season'].classes_:
            raise ValueError(f"Season '{season}' not recognized. Available: {list(label_encoders['season'].classes_)}")
        encoded_season = label_encoders['season'].transform([season])[0]
        df_reference_filtered['season'] = encoded_season # Overwrite with user's season

    # Now, ensure 'district_encoded' is present and correct
    original_district_name_for_encoding = df_reference_filtered['district'].values[0]
    if original_district_name_for_encoding.strip().lower() not in [k.strip().lower() for k in district_encoding_map.keys()]:
        # Handle new districts not in the training data
        df_reference_filtered['district_encoded'] = np.mean(list(district_encoding_map.values()))
    else:
        # Find the correct key from district_encoding_map, preserving its case
        matched_key = next((k for k in district_encoding_map.keys() if k.strip().lower() == original_district_name_for_encoding.strip().lower()), None)
        df_reference_filtered['district_encoded'] = district_encoding_map[matched_key]
        
    input_df_features = df_reference_filtered[feature_columns]

    input_scaled = scaler.transform(input_df_features)
    
    probabilities = model.predict_proba(input_scaled)[0]
    top_n_indices = np.argsort(probabilities)[::-1][:n]
    
    top_n_crops = [(label_encoders['Crop'].inverse_transform([idx])[0], probabilities[idx]) for idx in top_n_indices]
    
    return top_n_crops

def suggest_top_n_exotic_crops(features, exotic_df, n=2):
    crop_scores = []

    for _, row in exotic_df.iterrows():
        try:
            crop_name = row['Crop']
            temp_min = float(row['Temperature ©'])
            temp_max = float(row['Unnamed: 15'])
            ph = float(row['pH level'])
            humidity_min = float(row['Humidity (%)'])
            humidity_max = float(row['Unnamed: 17'])

            score = 0
            # Temperature score
            if not (temp_min <= features['temperature_2m_min'] <= temp_max):
                score += min(abs(features['temperature_2m_min'] - temp_min), abs(features['temperature_2m_min'] - temp_max))
            
            # pH score
            if not (ph - 0.25 <= features['ph'] <= ph + 0.25): # Smaller tolerance for pH
                score += abs(features['ph'] - ph) * 2 # Increase penalty for pH deviation

            # Humidity score
            if not (humidity_min <= features['relative_humidity_2m'] <= humidity_max):
                score += min(abs(features['relative_humidity_2m'] - humidity_min), abs(features['relative_humidity_2m'] - humidity_max))
            
            # Introduce a small random perturbation to scores to break ties
            score += np.random.uniform(0, 0.001) # Small random noise to break ties

            crop_scores.append((crop_name, score))
        except Exception as e:
            # print(f"Skipping exotic crop due to error: {e} for row: {row.get('Crop', 'Unknown')}")
            continue
    
    crop_scores.sort(key=lambda x: x[1])
    top_n_exotic_crops = [crop for crop, score in crop_scores[:n]]

    return top_n_exotic_crops if top_n_exotic_crops else ["No suitable exotic crop found"]

# --- User inputs (example) ---
user_month = 7
user_season = 'Kharif'
user_district = 'Akola' 

# --- Make predictions ---
print(f"\nPredicting for {user_district}, {user_season} season, Month {user_month}:")

# Predict top 3 traditional crops
top_3_traditional_crops = predict_top_n_crops(
    month=user_month,
    season=user_season,
    district=user_district,
    df_reference=original_df, # Use original_df for features
    label_encoders=label_encoders,
    scaler=scaler,
    model=ensemble,
    feature_columns=feature_columns,
    district_encoding_map=district_mean_encoding,
    n=3
)

# Extract feature values for exotic suggestion
# Ensure 'district' and 'season' columns are in original_df correctly before filtering
# This filter is based on the original string values of district and the *encoded* season value

# Make sure original_df has the season encoded before filtering for feature_row
# This ensures consistency whether models are loaded or trained
if original_df['season'].dtype == object: # Check if season is still string
    original_df['season'] = label_encoders['season'].transform(original_df['season'].str.strip())


feature_row = original_df[
    (original_df['district'].str.strip().str.lower() == user_district.strip().lower()) &
    (original_df['season'] == label_encoders['season'].transform([user_season.strip()])[0])
].head(1)

if not feature_row.empty:
    # --- New: Get pH from the mean encoding ---
    # Ensure the key for ph_mean_encoding is a tuple of (district_name, encoded_season)
    district_name_for_ph = feature_row['district'].values[0]
    encoded_season_for_ph = feature_row['season'].values[0] # This is already encoded
    
    # Check if the exact (district, season) combination exists in ph_mean_encoding
    if (district_name_for_ph, encoded_season_for_ph) in ph_mean_encoding:
        calculated_ph = ph_mean_encoding[(district_name_for_ph, encoded_season_for_ph)]
    else:
        # Fallback if specific district-season pH not found
        # You might want a more sophisticated fallback (e.g., district average, global average)
        print(f"Warning: pH for district '{user_district}' and season '{user_season}' not found. Using global average pH.")
        calculated_ph = np.mean(list(ph_mean_encoding.values())) # Fallback to global average pH

    exotic_input_features = {
        'temperature_2m_min': feature_row['temperature_2m_min'].values[0],
        'relative_humidity_2m': feature_row['relative_humidity_2m'].values[0],
        'ph': calculated_ph  # Use the calculated pH
    }

    top_2_exotic_crops = suggest_top_n_exotic_crops(exotic_input_features, exotic_df, n=2)
else:
    top_2_exotic_crops = ["No suitable exotic crop found (input row missing for features)"]

print("\n--- Prediction Results ---")
print("Top 3 Traditional Crops and their probabilities:")
for crop, prob in top_3_traditional_crops:
    print(f"- {crop}")

print("\nTop 2 Exotic Crops:")
for crop in top_2_exotic_crops:
    print(f"- {crop}")