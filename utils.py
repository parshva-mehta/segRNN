import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def calculate_distances(latlongs):
   num_stations = len(latlongs)
   distances = np.zeros((num_stations, num_stations))
   for i, coord1 in enumerate(latlongs):
      for j, coord2 in enumerate(latlongs):
         # Calculate Euclidean distance for normalized coordinates
         distances[i, j] = np.linalg.norm(np.array(coord1) - np.array(coord2))
   return distances

def handle_missing_values(df, continuous_cols):
   # Step 3: Handle missing values in continuous variables

   # Identify columns to drop due to high NaN count
   nan_threshold = df.shape[0] / 2                     # Remove columns with more than 50% missing values
   print(f"nan thresh is {nan_threshold}")
   bad_columns = [col for col in df.columns if df[col].isnull().sum() >= nan_threshold]
   print(f"bad columns are {bad_columns}")

   # Add less relevant and irrelevant features to the removal list
   irrelevant_features = [
      'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr',  # Fully NaN
      'skyl1', 'skyl2', 'skyl3', 'skyl4',  # Less relevant (Sky level altitudes)
      'skyc1', 'skyc2', 'skyc3', 'skyc4',  # Less relevant (Sky coverage)
      'wxcodes',  # Categorical, redundant with precipitation/visibility
      'metar'  # Text format, unusable directly
   ]

   # Combine both lists and ensure no duplicates
   columns_to_remove = list(set(bad_columns + irrelevant_features))
   df.drop(columns=columns_to_remove, inplace=True)

   # Update continuous columns to exclude removed columns
   continuous_cols = list(set(continuous_cols) - set(columns_to_remove))
   print(f"{len(continuous_cols)} remaining continuous columns: {continuous_cols}")


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
   continuous_cols = ['tmpf', 'dwpf', 'relh', 'feel', 'drct', 'sknt', 'gust',
                     'peak_wind_gust', 'peak_wind_drct', 'alti', 'mslp', 'vsby',
                     'p01i', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr',
                     'skyl1', 'skyl2', 'skyl3', 'skyl4', 'snowdepth', 'peak_wind_time']

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
