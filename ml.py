import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


soil_data = pd.read_csv('soil_data.csv')
weather_data = pd.read_csv('weather_data.csv')
yield_data = pd.read_csv('crop_yield.csv')


combined_data = pd.merge(soil_data, weather_data, on=['region', 'year'])
combined_data = pd.merge(combined_data, yield_data, on=['region', 'year'])

# Data Cleaning
imputer = KNNImputer(n_neighbors=5)
clean_data = imputer.fit_transform(combined_data.select_dtypes(include=[np.number]))

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(clean_data)

# Time-Series Reshaping
sequence_length = 2  # Adjust for testing with smaller data
sequences = []
targets = []

for i in range(len(normalized_data) - sequence_length):
    sequences.append(normalized_data[i:i+sequence_length, :-1])  # Features
    targets.append(normalized_data[i+sequence_length, -1])       
sequences = np.array(sequences)
targets = np.array(targets)

# CNN-LSTM Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sequence_length, sequences.shape[2])))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training
X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=2, callbacks=[early_stopping])

# Evaluation
predictions = model.predict(X_val)
rmse = mean_squared_error(y_val, predictions, squared=False)
r2 = r2_score(y_val, predictions)

print(f"RMSE: {rmse}, R²: {r2}")
