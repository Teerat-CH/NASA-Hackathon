import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

st.header("Kp Index Forecasting Model")

# Load dataset and model
data = pd.read_csv("Final_Data_for_Model_training.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.date  # Convert to date

model = tf.keras.models.load_model('DSCOVR')

# Split the data
features = data.iloc[:, 2:].values
target = data['Kp'].values

X_train, X_test, y_train, y_test = train_test_split(data.drop('Kp', axis=1), data['Kp'], test_size=0.2, random_state=42)
test_dates = X_test['Timestamp'].values

# Extract only the features without timestamp for the LSTM model
X_train_features = X_train.drop('Timestamp', axis=1).values
X_test_features = X_test.drop('Timestamp', axis=1).values

# Reshape the data
X_train_reshaped = X_train_features.reshape(X_train_features.shape[0], X_train_features.shape[1], 1)
X_test_reshaped = X_test_features.reshape(X_test_features.shape[0], X_test_features.shape[1], 1)

if st.checkbox('Show Dataset'):
    st.write(data)

st.subheader("Please input relevant features for forecasting!")

# Allow the user to select a date
selected_date = st.selectbox('Select Date', test_dates)
selected_time_index = np.where(test_dates == selected_date)[0][0]

if st.button('Make Prediction'):
    # Get the features corresponding to the selected date
    selected_features = X_test_features[selected_time_index].reshape(1, -1, 1)

    # Predict using the LSTM model
    prediction = model.predict(selected_features)

    # Display the specific KP prediction for the selected time
    st.write(f"Predicted KP Index for {selected_date} is: {prediction[0][0]:.2f}")

    st.write(f"Thank you! I hope you found this useful.")
    st.write(f"Even though our current app can only forecast historical data, we are determined to connect to data in the future to predict real-time Kp index!")

    
# Display the graph
y_pred_all = model.predict(X_test_reshaped)
chart_data = pd.DataFrame({
    'Real KP Values': y_test.values,
    'Predicted KP Values': y_pred_all.flatten()
})
st.line_chart(chart_data)
