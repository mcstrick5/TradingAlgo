#Saves the model and encoder/scaler
#use this code to implement the updates Richard suggested and Richard only 
#Normalize the features
#Use the next day close as the target
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold, GridSearchCV
import statsmodels.api as sm
from scipy.stats import zscore
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.feature_extraction import MinimalFCParameters
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import mutual_info_regression
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optuna.visualization import (
    plot_contour,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_edf,
    plot_optimization_history,
    plot_timeline,
    plot_rank,
)
import joblib
import os

path = r'C:\Users\mclau\Documents\Backtest\Code\data\EUR_USD_D.csv'

# Read the data
data = pd.read_csv(path, parse_dates=['time'], index_col='time')

# Select the required columns
columns = ['volume', 'mid_o', 'mid_h', 'mid_l', 'mid_c',
           'bid_o', 'bid_h', 'bid_l', 'bid_c', 'ask_o', 'ask_h', 'ask_l', 'ask_c']
data = data[columns]

# Filter data starting from 2009
data = data[data.index >= '2009-01-01']

# Reset the index to make 'time' a column
data.reset_index(inplace=True)

# Ensure all columns are numeric and convert all columns to floats
data[columns] = data[columns].apply(pd.to_numeric, errors='coerce').astype(float)

# Handle missing values
data.dropna(inplace=True)

# Split data into training and testing sets
train_data = data[data['time'] < '2022-01-01']  # Adjust date as needed for splitting
test_data = data[data['time'] >= '2022-01-01']  # Adjust date as needed for splitting

# Normalize the training data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[columns])

# Train the autoencoder on the training data
input_dim = train_data_scaled.shape[1]
encoding_dim = 280  # Number of features to learn

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the autoencoder with early stopping on the training data
history = autoencoder.fit(train_data_scaled, train_data_scaled, epochs=100, batch_size=32, shuffle=False, validation_split=0.2, callbacks=[early_stopping])

# Extract the encoder part to use it for feature extraction
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Extract features from the training and testing data
train_encoded_data = encoder_model.predict(train_data_scaled)
test_data_scaled = scaler.transform(test_data[columns])
test_encoded_data = encoder_model.predict(test_data_scaled)

# Convert the encoded data to DataFrames
train_encoded_df = pd.DataFrame(train_encoded_data, columns=[f'feature_{i+1}' for i in range(encoding_dim)])
test_encoded_df = pd.DataFrame(test_encoded_data, columns=[f'feature_{i+1}' for i in range(encoding_dim)])

# Add the time column back
train_encoded_df['time'] = train_data['time'].values
test_encoded_df['time'] = test_data['time'].values

# Merge with the original data
train_data_with_features = pd.concat([train_data.reset_index(drop=True), train_encoded_df], axis=1)
test_data_with_features = pd.concat([test_data.reset_index(drop=True), test_encoded_df], axis=1)

# Now you can proceed with additional feature calculation on train_data_with_features and test_data_with_features
# Adding common OHLCV features manually
def calculate_features(df):
    # Simple Moving Averages (SMA)
    df['SMA_10'] = df['mid_c'].rolling(window=10).mean()
    df['SMA_30'] = df['mid_c'].rolling(window=30).mean()
    
    # Exponential Moving Averages (EMA)
    df['EMA_10'] = df['mid_c'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['mid_c'].ewm(span=30, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['mid_c'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['mid_c'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['mid_c'].rolling(window=20).std()
    
    # Relative Strength Index (RSI)
    delta = df['mid_c'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['mid_c'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['mid_c'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Average True Range (ATR)
    df['H-L'] = df['mid_h'] - df['mid_l']
    df['H-PC'] = np.abs(df['mid_h'] - df['mid_c'].shift(1))
    df['L-PC'] = np.abs(df['mid_l'] - df['mid_c'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Candlestick Patterns
    df['Bullish_Engulfing'] = np.where((df['mid_o'] < df['mid_c'].shift(1)) & (df['mid_c'] > df['mid_o'].shift(1)) &
                                       (df['mid_c'] > df['mid_o']) & (df['mid_o'] < df['mid_c'].shift(1)), 1, 0)
    df['Bearish_Engulfing'] = np.where((df['mid_o'] > df['mid_c'].shift(1)) & (df['mid_c'] < df['mid_o'].shift(1)) &
                                       (df['mid_c'] < df['mid_o']) & (df['mid_o'] > df['mid_c'].shift(1)), 1, 0)
    
    return df

# Calculate and add features to the DataFrame
train_data_with_features = calculate_features(train_data_with_features)

# Handle missing values after feature creation
train_data_with_features.dropna(inplace=True)

# Check the number of features
print(f"Number of features after adding technical indicators: {train_data_with_features.shape[1]}")

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Check if early stopping was triggered
trained_epochs = len(history.history['loss'])
max_epochs = 100  # The maximum number of epochs you set

if trained_epochs < max_epochs:
    print(f"Early stopping was triggered at epoch {trained_epochs}.")
else:
    print(f"Early stopping was not triggered. Model trained for {max_epochs} epochs.")

# Compute mutual information
X = train_data_with_features.drop(columns=['time', 'mid_c'])  # Excluding the target variable 'mid_c' and 'time'
y = train_data_with_features['mid_c']

mi = mutual_info_regression(X, y)

# Create a DataFrame to display the mutual information scores
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Plot the mutual information scores
plt.figure(figsize=(10, 8))
sns.barplot(x='Mutual Information', y='Feature', data=mi_df)
plt.title('Mutual Information Scores')
plt.show()

# Display mutual information scores for interpretation
print(mi_df)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(train_data_with_features, test_size=0.2, shuffle=False)

# Print train/test dates
print(f"Train start: {train_data['time'].iloc[0]}, Train end: {train_data['time'].iloc[-1]}")
print(f"Test start: {test_data['time'].iloc[0]}, Test end: {test_data['time'].iloc[-1]}")

# Visualization setup
plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(train_data['time'], train_data['mid_c'], color='blue', label='Train Data')

# Plot testing data
plt.plot(test_data['time'], test_data['mid_c'], color='red', label='Test Data')

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Train/Test Data Visualization')
plt.legend(loc='upper left')
plt.show()

# Define features and target
features = train_data.drop(columns=['mid_c', 'time']).columns.tolist()

# Initialize variables for collecting results
all_y_test = []
all_y_pred = []
all_time = []

# Training and prediction
X_train = train_data[features]
y_train = train_data['mid_c']
X_test = test_data[features]
y_test = test_data['mid_c']

# Define the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Collect results
all_y_test.extend(y_test.values)
all_y_pred.extend(y_pred)
all_time.extend(test_data['time'].values)

# Adjust lengths to match
all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)
all_time = np.array(all_time)

# Generate signals
signals = np.where(all_y_pred[1:] > all_y_test[:-1], 1, -1)  # 1 for buy, -1 for sell

# Calculate accuracy of signals
true_directions = np.sign(np.diff(all_y_test))
accuracy = accuracy_score(true_directions, signals)
print(f"Signal accuracy: {accuracy * 100:.2f}%")

# Plot predictions vs true values
plt.figure(figsize=(10, 8))
plt.scatter(all_y_test, all_y_pred, alpha=0.3)
plt.plot([min(all_y_test), max(all_y_test)], [min(all_y_test), max(all_y_test)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('XGBoost Predictions vs True Values')
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(all_time, all_y_test, label='Actual Prices', color='blue')
plt.plot(all_time, all_y_pred, label='Predicted Prices', color='orange')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend(loc='upper left')
plt.show()

# Create the specified directory to save model artifacts
save_dir = r'C:\Users\mclau\Documents\Backtest\Code\live_bot'
os.makedirs(save_dir, exist_ok=True)

# After training the autoencoder, save it
autoencoder.save(os.path.join(save_dir, 'autoencoder_model.keras'))
encoder_model.save(os.path.join(save_dir, 'encoder_model.keras'))


# After fitting the XGBoost model, save it
joblib.dump(model, os.path.join(save_dir, 'xgboost_model.pkl'))

# Save the scaler
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

# Save feature names for reference
joblib.dump(features, os.path.join(save_dir, 'feature_names.pkl'))

# Function to load the models and make predictions on new data
def load_models_and_predict(new_data_path, models_dir=r'C:\Users\mclau\Documents\Backtest\Code\live_bot'):
    """
    Load saved models and make predictions on new data
    
    Parameters:
    new_data_path (str): Path to the new data CSV file
    models_dir (str): Directory where model artifacts are stored
    
    Returns:
    DataFrame: Original data with predictions
    """
     # Load saved components
    loaded_scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    loaded_encoder = load_model(os.path.join(models_dir, 'encoder_model.keras'))
    loaded_xgb = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    
    # Read new data
    new_data = pd.read_csv(new_data_path, parse_dates=['time'], index_col='time')
    
    # Select the required columns (same as in training)
    columns = ['volume', 'mid_o', 'mid_h', 'mid_l', 'mid_c',
               'bid_o', 'bid_h', 'bid_l', 'bid_c', 'ask_o', 'ask_h', 'ask_l', 'ask_c']
    new_data = new_data[columns]
    
    # Reset index to make 'time' a column
    new_data.reset_index(inplace=True)
    
    # Scale the data
    new_data_scaled = loaded_scaler.transform(new_data[columns])
    
    # Extract features using the encoder
    new_encoded_data = loaded_encoder.predict(new_data_scaled)
    
    # Convert encoded data to DataFrame
    encoding_dim = new_encoded_data.shape[1]
    new_encoded_df = pd.DataFrame(new_encoded_data, 
                                  columns=[f'feature_{i+1}' for i in range(encoding_dim)])
    
    # Add time column back
    new_encoded_df['time'] = new_data['time'].values
    
    # Merge with original data
    new_data_with_features = pd.concat([new_data.reset_index(drop=True), new_encoded_df], axis=1)
    
    # Calculate technical indicators
    new_data_with_features = calculate_features(new_data_with_features)
    
    # Handle missing values
    new_data_with_features.dropna(inplace=True)
    
    # Ensure we have all required features (some may be missing in new data)
    for feature in feature_names:
        if feature not in new_data_with_features.columns:
            new_data_with_features[feature] = 0
    
    # Make predictions
    X_new = new_data_with_features[feature_names]
    predictions = loaded_xgb.predict(X_new)
    
    # Add predictions to the DataFrame
    new_data_with_features['predicted_price'] = predictions
    
    # Generate signals (1 for buy, -1 for sell)
    signals = np.where(predictions[1:] > new_data_with_features['mid_c'].values[:-1], 1, -1)
    signals = np.append(0, signals)  # Add 0 for the first day (no signal)
    new_data_with_features['signal'] = signals
    
    return new_data_with_features

