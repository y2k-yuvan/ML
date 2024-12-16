import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

def main():
    st.title("Crop Yield Prediction with CNN-LSTM")

    # File Uploads
    st.sidebar.header("Upload Data")
    soil_file = st.sidebar.file_uploader("Upload Soil Data (CSV)", type="csv")
    weather_file = st.sidebar.file_uploader("Upload Weather Data (CSV)", type="csv")
    yield_file = st.sidebar.file_uploader("Upload Crop Yield Data (CSV)", type="csv")

    if soil_file and weather_file and yield_file:
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
        model.add(Flatten())
        model.add(LSTM(50, return_sequences=False))
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
                rmse = mean_squared_error(y_val, predictions, squared=False)
                r2 = r2_score(y_val, predictions)

                st.success("Model Training Complete!")
                st.write(f"**RMSE:** {rmse}")
                st.write(f"**R² Score:** {r2}")

if __name__ == "__main__":
    main()
