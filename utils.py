import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def calculate_distances(latlongs):
   num_stations = len(latlongs)
   distances = np.zeros((num_stations, num_stations))
   for i, coord1 in enumerate(latlongs):
      for j, coord2 in enumerate(latlongs):
         # Calculate Euclidean distance for normalized coordinates
         distances[i, j] = np.linalg.norm(np.array(coord1) - np.array(coord2))
   return distances

def handle_missing_values(df, continuous_cols):
   # Use only the specified 10 columns
   continuous_cols = ['feel', 'relh', 'tmpf', 'vsby', 'sknt', 'mslp', 'p01i', 'alti', 'dwpf', 'drct']

   # Ensure only these columns are retained
   df = df[continuous_cols + ['station']]

   print(f"Using fixed continuous columns: {continuous_cols}")

   # Apply linear interpolation within each station group using transform
   df[continuous_cols] = df.groupby('station')[continuous_cols].transform(
      lambda group: group.interpolate(method='linear')
   )

   # Handle any remaining missing values with forward and backward fill using transform
   df[continuous_cols] = df.groupby('station')[continuous_cols].transform(
      lambda group: group.ffill().bfill()
   )
   return df, continuous_cols

def preprocess_and_save_data(input_path, normalize=True):
   print('new function!')
   base_name = os.path.basename(input_path)  # Extracts 'JRB.csv'
   file_name, file_ext = os.path.splitext(base_name)  # Splits into 'JRB' and '.csv'
   output_path = os.path.join(os.path.dirname(input_path), f"{file_name}_processed{file_ext}")

   # if os.path.exists(output_path):
   #    print(f"The processed file already exists at {output_path}. Skipping processing.")
   #    return

   # Step 1 read
   df = pd.read_csv(input_path, low_memory=False)
   df['valid'] = pd.to_datetime(df['valid'])
   df = df.sort_values(by=['station', 'valid']).reset_index(drop=True)

   # Step 2: Replace placeholders with np.nan in continuous columns
   continuous_cols = ['feel', 'relh', 'tmpf', 'vsby', 'sknt', 'mslp', 'p01i', 'alti', 'dwpf', 'drct']

   # List of placeholders to replace
   placeholders = ['M', 'T', '', 'NaN', 'NULL', 'None']

   # Replace placeholders with np.nan
   df[continuous_cols] = df[continuous_cols].replace(placeholders, np.nan).astype(str)

   # Convert continuous columns to numeric, coercing errors to np.nan
   for col in continuous_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce')

   print("Missing values in continuous columns before processing:")
   print(df[continuous_cols].isnull().sum())

   df, continuous_cols = handle_missing_values(df, continuous_cols)

   # Verify missing values are filled
   print("Missing values in continuous columns after processing:")
   print(df[continuous_cols].isnull().sum())

   # Step 5: Feature scaling
   # List of all features (excluding 'valid' and 'metar')
   feature_cols = continuous_cols #+ categorical_cols
   print(df[feature_cols].shape)

   if normalize:
      print("normalizing...")
      # Initialize the scaler
      scaler = StandardScaler()
      # Fit and transform the features (not yet)
      df[feature_cols] = scaler.fit_transform(df[feature_cols])

   print(f"saving csv to {output_path}")
   df.to_csv(output_path, index=False)

   return feature_cols
