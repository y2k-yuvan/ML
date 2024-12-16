import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

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

            # Data Preprocessing
            st.write("### Combining and Preprocessing Data")
            combined_data = pd.merge(soil_data, weather_data, on=['region', 'year'])
            combined_data = pd.merge(combined_data, yield_data, on=['region', 'year'])

            # Handle missing data
            imputer = KNNImputer(n_neighbors=5)
            clean_data = imputer.fit_transform(combined_data.select_dtypes(include=[np.number]))

            # Normalize the data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(clean_data)

            st.write("Data successfully preprocessed!")

            # Sequence Creation (keep 6 features)
            sequence_length = st.sidebar.slider("Sequence Length", 1, 10, value=2)
            sequences = []
            targets = []

            for i in range(len(normalized_data) - sequence_length):
                sequences.append(normalized_data[i:i + sequence_length, :-1])  # Features (6 features)
                targets.append(normalized_data[i + sequence_length, -1])  # Target (Crop Yield)
            
            sequences = np.array(sequences)
            targets = np.array(targets)

            # Model Definition - Working with 6 features
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sequence_length, sequences.shape[2])))
            model.add(MaxPooling1D(pool_size=1))
            model.add(LSTM(50, return_sequences=False))  # Directly using LSTM without Flatten
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
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
                    rmse = np.sqrt(mse)  # Manually calculate RMSE

                    # Check for constant target values to avoid NaN R²
                    if np.var(y_val) == 0:
                        r2 = float('nan')
                        st.warning("R² Score is undefined because the target values are constant.")
                    else:
                        r2 = r2_score(y_val, predictions)

                    st.success("Model Training Complete!")
                    st.write(f"**RMSE:** {rmse}")
                    st.write(f"**R² Score:** {r2}")

            # Prediction Input (takes 6 features)
            st.write("### Predict Crop Yield")
            region = st.number_input("Region", min_value=1, max_value=3, value=1)
            year = st.number_input("Year", min_value=2023, max_value=2024, value=2023)
            soil_type = st.selectbox("Soil Type", ["Silty", "Clayey", "Loamy", "Peaty"])
            ph_level = st.number_input("pH Level", value=6.5)
            moisture_content = st.number_input("Moisture Content", value=22)
            organic_matter = st.number_input("Organic Matter", value=2.5)
            fertility_level = st.number_input("Fertility Level", value=8)

            if st.button("Predict"):
                # Map categorical variables (e.g., soil_type) to numerical values
                soil_type_map = {"Silty": 1, "Clayey": 2, "Loamy": 3, "Peaty": 4}
                soil_type_num = soil_type_map.get(soil_type, 1)

                # Prepare input features (6 features)
                input_features = np.array([[region, year, soil_type_num, ph_level, moisture_content, organic_matter, fertility_level]])
                input_features = scaler.transform(input_features)  # Normalize input

                # Predict using the trained model
                prediction = model.predict(input_features)
                st.write(f"Predicted Crop Yield: {prediction[0][0]:.2f} kg/hectare")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
