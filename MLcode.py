import streamlit as st
import pandas as pd
import numpy as np

# Check for required libraries
try:
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    st.error(f"A required library is missing: {e}. Please install it using 'pip install scikit-learn tensorflow pandas numpy'.")
    st.stop()

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

            imputer = KNNImputer(n_neighbors=5)
            clean_data = imputer.fit_transform(combined_data.select_dtypes(include=[np.number]))

            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(clean_data)

            st.write("Data successfully preprocessed!")

            # Sequence Creation
            sequence_length = st.sidebar.slider("Sequence Length", 1, 10, value=2)
            sequences = []
            targets = []

            for i in range(len(normalized_data) - sequence_length):
                sequences.append(normalized_data[i:i + sequence_length, :-1])  # Features
                targets.append(normalized_data[i + sequence_length, -1])       
            sequences = np.array(sequences)
            targets = np.array(targets)

            # Model Definition
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

            # User input for prediction (request exactly 8 inputs)
            st.subheader("Enter values for prediction")

            # List of 8 features expected by the model
            all_features = [
                'Temperature', 'Rainfall', 'Soil Quality', 'Humidity', 
                'Region', 'Year', 'Crop Type', 'Fertilizer Usage'
            ]

            # Create a dictionary for user inputs
            inputs = {}

            for feature in all_features:
                inputs[feature] = st.number_input(f"Enter {feature} value", value=0.0)  # Default value

            # Prepare the input data for prediction
            input_data = np.array([list(inputs.values())])

            # Ensure the input data has the correct shape (only 7 features, drop the last one)
            input_data_7_features = input_data[:, :-1]  # Slice to use the first 7 features, assuming the 8th is the target

            # Scale the input data (since the model expects normalized input)
            input_data_scaled = scaler.transform(input_data_7_features)

            # Ensure input data is in the correct shape (similar to training sequence)
            input_data_sequence = np.expand_dims(input_data_scaled, axis=1)  # Expanding to match the sequence input (1, sequence_length, 7)

            # Predict crop yield when the button is clicked
            if st.button("Predict Crop Yield"):
                prediction = model.predict(input_data_sequence)
                st.write(f"Predicted Crop Yield: {prediction[0][0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()

