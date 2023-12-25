import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load the data
data = pd.read_csv('data.csv')

# Exclude columns 'mstat', 'stdate', 'rank'
data_subset = data.drop(['mstat', 'stdate', 'rank'], axis=1)

# Display the image on top of the page
image = plt.imread('image.png')
st.image(image, use_column_width=True)

# Create sliders for each feature
col1, col2 = st.columns(2)
sliders = {}
for i, feature in enumerate(data_subset.columns):
    min_val = float(data_subset[feature].min())
    max_val = float(data_subset[feature].max())
    if i % 2 == 0:
        sliders[feature] = col1.slider(f'{feature}: {min_val:.2f} - {max_val:.2f}', min_val, max_val)
    else:
        sliders[feature] = col2.slider(f'{feature}: {min_val:.2f} - {max_val:.2f}', min_val, max_val)

# Prepare the input data for prediction
input_data = pd.DataFrame([sliders])

# Fit the Isolation Forest model
model = IsolationForest(contamination=0.05)
model.fit(data_subset)

# Predict the anomaly scores for the input data
anomaly_scores = model.decision_function(input_data)

# Set a custom threshold for anomaly detection
custom_threshold = -0.2

# Determine if the input data is an outlier or not
if anomaly_scores[0] < custom_threshold:
    result = 'Outlier'
else:
    result = 'Pass'

# Calculate the accuracy of the model
y_pred = model.predict(data_subset)
y_true = pd.Series([1] * len(data_subset))
accuracy = (y_pred == y_true).mean()

# Plot scattergram of all data points
plt.figure(figsize=(8, 6))

# Plot original data with hue
sns.scatterplot(data=data_subset, x='age', y='wyr', hue='gender', palette='viridis', marker='o')

# Highlight outliers
outliers = data_subset[model.predict(data_subset) == -1]
sns.scatterplot(data=outliers, x='age', y='wyr', hue='gender', palette='Reds', marker='x')

plt.xlabel('Age')
plt.ylabel('WYR')
plt.title('Scattergram of Data Points')
plt.legend()

# Display the scatter plot
st.pyplot(plt)

# Display the result and accuracy
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<h1><b>Prediction: {result}</b></h1>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<h1><b>Outlier Score: {anomaly_scores[0]:.2f}</b></h1>", unsafe_allow_html=True)
st.markdown(f"<h1><b>Accuracy: {accuracy:.4f}</b></h1>", unsafe_allow_html=True)