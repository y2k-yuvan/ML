import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("Crop Yield Prediction with CNN-LSTM")

    # File Uploads
    st.sidebar.header("Upload Data")
    soil_file = st.sidebar.file_uploader("Upload Soil Data (CSV)", type="csv")
    weather_file = st.sidebar.file_uploader("Upload Weather Data (CSV)", type="csv")
    yield_file = st.sidebar.file_uploader("Upload Crop Yield Data (CSV)", type="csv")

    if soil_file and weather_file and yield_file:
        try:
            # Load data
            soil_data = pd.read_csv(soil_file)
            weather_data = pd.read_csv(weather_file)
            yield_data = pd.read_csv(yield_file)

            st.write("### Soil Data Sample")
            st.dataframe(soil_data.head())

            st.write("### Weather Data Sample")
            st.dataframe(weather_data.head())

            st.write("### Yield Data Sample")
            st.dataframe(yield_data.head())

            # Combine the datasets on 'region' and 'year'
            combined_data = pd.merge(soil_data, weather_data, on=['region', 'year'])
            combined_data = pd.merge(combined_data, yield_data, on=['region', 'year'])

            # Selecting only the features relevant to the prediction (7 features)
            features = ['temperature', 'humidity', 'ph_level', 'moisture_content', 
                        'organic_matter', 'fertility_level', 'salinity']

            # Imputation for missing values
            imputer = KNNImputer(n_neighbors=5)
            clean_data = imputer.fit_transform(combined_data[features])

            # Normalizing only the selected features
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(clean_data)

            # Model Setup
            sequence_length = 1  # Keeping sequence length as 1 for simplicity
            sequences = []
            targets = []

            for i in range(len(normalized_data) - sequence_length):
                sequences.append(normalized_data[i:i + sequence_length, :-1])  # Features (7 features excluding the yield)
                targets.append(normalized_data[i + sequence_length, -1])  # Yield (target)
            sequences = np.array(sequences)
            targets = np.array(targets)

            # Model Definition
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sequence_length, sequences.shape[2])))
            model.add(MaxPooling1D(pool_size=1))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))  # Output layer for predicting yield
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train-Test Split
            X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=0.2, random_state=42)

            if st.sidebar.button("Train Model"):
                with st.spinner("Training the model, please wait..."):
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
                    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=2, callbacks=[early_stopping])

                    # Evaluation
                    predictions = model.predict(X_val)
                    mse = mean_squared_error(y_val, predictions)
                    rmse = np.sqrt(mse)

                    r2 = r2_score(y_val, predictions)

                    st.success("Model Training Complete!")
                    st.write(f"**RMSE:** {rmse}")
                    st.write(f"**R² Score:** {r2}")

            # User input for prediction
            st.sidebar.header("Enter Prediction Input")
            temp = st.sidebar.number_input("Temperature (°C)", value=28.4)
            humidity = st.sidebar.number_input("Humidity (%)", value=75)
            ph = st.sidebar.number_input("pH Level", value=6.8)
            moisture = st.sidebar.number_input("Moisture Content (%)", value=22)
            organic_matter = st.sidebar.number_input("Organic Matter (%)", value=3.0)
            fertility = st.sidebar.number_input("Fertility Level (1-10)", value=7)
            salinity = st.sidebar.number_input("Salinity (%)", value=0.2)

            # Prepare user input for prediction
            input_data = np.array([[temp, humidity, ph, moisture, organic_matter, fertility, salinity]])
            input_data = scaler.transform(input_data)  # Normalize the user input

            # Reshape input for prediction
            input_data = input_data.reshape((1, 1, 7))  # 1 sample, 1 timestep, 7 features (excluding crop type)

            if st.sidebar.button("Predict Yield"):
                prediction = model.predict(input_data)
                st.write(f"Predicted Crop Yield: {prediction[0][0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
