import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


st.header("Kp Index Forecasting Model")

# Load dataset and model
data = pd.read_csv("Final_Data_for_Model_training.csv")
model = tf.keras.models.load_model('DSCOVR')
#split the data
features = data.iloc[:, 2:].values
target = data['Kp'].values
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#reshape the data
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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

# Display the graph
y_pred_all = model.predict(X_test_reshaped)  # Assuming X_test_reshaped is available
chart_data = pd.DataFrame({
    'Real KP Values': y_test,  # Assuming y_test is globally available
    'Predicted KP Values': y_pred_all.flatten()
})
st.line_chart(chart_data)