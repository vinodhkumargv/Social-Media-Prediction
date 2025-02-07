#!/usr/bin/env python
# coding: utf-8

# ## 1.Data Loading 

# In[300]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
data = pd.read_csv("C:/Users/Vinodh/Desktop/Major Project/SocialMedia_Mental_Health.csv")

# Check for missing values
#print(data.isnull().sum())


# Display the first few rows
data.head()



# ## 2.Data Cleaning

# In[301]:


# Handle missing values (if any)
data = data.dropna()  # or use data.fillna() for imputation

# Convert categorical variables to numerical (if needed)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1, 'Trans': 2})  # -1 often represents 'Other' or 'Unknown'
data['Relationship Status'] = data['Relationship Status'].map({'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3})


# Check the cleaned data
data.head()


# ## 3.EDA

# In[302]:


plt.figure(figsize=(10, 6))
sns.histplot(data['What is your age?'], bins=10, kde=True, color="royalblue", edgecolor="black", alpha=0.7)  # Add transparency
sns.kdeplot(data['What is your age?'], color="darkred", linewidth=2)  # Customize KDE line

# Age Distribution
plt.title("Age Distribution", fontsize=14, fontweight="bold", color="darkblue")
plt.xlabel("Age (Years)", fontsize=12, fontweight="bold")
plt.ylabel("Frequency", fontsize=12, fontweight="bold")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle="--", alpha=0.6)  # Add dashed gridlines

# Show the chart
plt.show()





# In[303]:


#Distibution of Time Spent on Social Media

plt.figure(figsize=(10, 6))
sns.histplot(data['What is the average time you spend on social media every day?'], bins=10, kde=True, color="purple")
plt.title("Distribution of Time Spent on Social Media", fontsize=14)
plt.xlabel("Time Spent (Hours per Day)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[304]:


# Time Spent on Social Media vs Distraction
# Step 1: Map Time Spent and Distraction Level to Numeric Values (if categorical)

time_mapping = {
    "Between 1 and 2 hours": 1.5,
    "Between 2 and 3 hours": 2.5,
    "Between 3 and 4 hours": 3.5,
    "Between 4 and 5 hours": 4.5,
    "More than 5 hours": 6,
    "Less than an Hour": 0.5
}


# Apply the mapping to the relevant columns
data['Time Spent on Social Media (Hours)'] = data['What is the average time you spend on social media every day?'].map(time_mapping)
#data['Distraction Level'] = data['How often do you get distracted by Social media when you are busy doing something?'].map(distraction_mapping)


data.rename(columns={
    #'What is the average time you spend on social media every day?': 'Time Spent on Social Media (Hours)',
    'How often do you get distracted by Social media when you are busy doing something?': 'Distraction Level'
}, inplace=True)


# Box Plot on Time Spent Vs Distraction

plt.figure(figsize=(12, 6))

# Create the box plot with the palette applied to the 'hue' variable
sns.boxplot(x='Time Spent on Social Media (Hours)', 
            y='Distraction Level', 
            data=data, 
            hue='Time Spent on Social Media (Hours)',  # Use 'hue' to differentiate categories
            palette="coolwarm",  # Color palette to visually distinguish categories
            showfliers=True)  # Show outliers

# Title and labels for clarity
plt.title("Time Spent on Social Media vs Distraction Levels", fontsize=16, fontweight="bold", color="darkblue")
plt.xlabel("Time Spent on Social Media (Hours per Day)", fontsize=14)
plt.ylabel("Distraction Level", fontsize=14)
plt.xticks(fontsize=12, rotation=45)  # Rotate x-axis labels for better visibility
plt.yticks(fontsize=12)
plt.grid(True)

# Remove legend if not required
plt.legend([],[], frameon=False)

# Show the plot
plt.tight_layout()  # Ensure everything fits well
plt.show()


# In[305]:


# EDA: Correlation Heatmap
plt.figure(figsize=(25, 18))
corr = data.corr(numeric_only=True)

# Increase the font size of the annotations (numbers inside the heatmap)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12})  # Adjust size as needed

# Increase the font size of the axis labels (x and y axis)
plt.xticks(fontsize=25)  # Adjust fontsize as needed
plt.yticks(fontsize=25)  # Adjust fontsize as needed

plt.title("Correlation Heatmap", fontsize=25)  # Adjust title fontsize
plt.xlabel("Features", fontsize=25) # Adjust x label fontsize
plt.ylabel("Features", fontsize=25) # Adjust y label fontsize

plt.show()





# In[306]:


# Define addiction score based on multiple factors
data['Addiction Score'] = (
    data['Time Spent on Social Media (Hours)'] +
    data['How often do you find yourself using Social media without a specific purpose?'] +
    #data['How often do you get distracted by Social media when you are busy doing something?'] +
    data['Do you feel restless if you haven\'t used Social media in a while?'] +
    data['On a scale of 1 to 5, how easily distracted are you?'] +
    data['On a scale of 1 to 5, how much are you bothered by worries?'] +
    data['Do you find it difficult to concentrate on things?'] +
    data['On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?'] +
    data['How often do you look to seek validation from features of social media?'] +
    data['How often do you feel depressed or down?'] +
    data['On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?'] +
    data['On a scale of 1 to 5, how often do you face issues regarding sleep?']
)



# EDA: Boxplot of Addiction Score by Gender
plt.figure(figsize=(8, 6))
#sns.boxplot(x='Gender', y='Addiction Score', data=data)
sns.histplot(x='Gender', data=data)
#sns.violinplot(x='Gender', y='Addiction Score', data=data) 
plt.title("Addiction Score by Gender")
plt.xlabel("Gender")
plt.xticks([0, 1, 2], ['Male', 'Female', 'Trans'])
plt.ylabel("Addiction Score")
plt.show()


# In[307]:


# EDA: Distribution of Addiction Score
plt.figure(figsize=(8, 6))
sns.histplot(data['Addiction Score'], kde=True, bins=20, color='blue')
plt.title("Distribution of Addiction Score")
plt.xlabel("Addiction Score")
plt.ylabel("Frequency")
plt.show()


# ## 4.Predictive Analysis

# In[308]:


from sklearn.impute import SimpleImputer

time_mapping = {
    "Between 1 and 2 hours": 1.5,
    "Between 2 and 3 hours": 2.5,
    "Between 3 and 4 hours": 3.5,
    "Between 4 and 5 hours": 4.5,
    "More than 5 hours": 6,
    "Less than an Hour": 0.5
}

#data.head()

# Apply the mapping to the relevant columns
data['Time Spent on Social Media (Hours)'] = data['What is the average time you spend on social media every day?'].map(time_mapping)
data.head()
# Select features and target variable
X = data[['What is your age?', 'Gender', 'Time Spent on Social Media (Hours)', 
          'How often do you find yourself using Social media without a specific purpose?']]
y = data['How often do you feel depressed or down?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# ## 5.Model Training

# In[309]:


import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import StandardScaler

#print(data.dtypes)

# Define Features and Target

X = data.drop(data.select_dtypes(include=['object']).columns, axis=1)  # Remove inplace=True
y = data['Addiction Score'] > 25

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model using pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")


# In[310]:


# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[311]:


# Feature Importance
plt.figure(figsize=(12, 6))
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# In[312]:


# ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


# In[313]:


# PCA for Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="coolwarm")
plt.title("PCA Projection of Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# In[314]:


# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
#kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis")
plt.title("K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()




# In[315]:


# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Social Media & Mental Health Dashboard", style={'textAlign': 'center'}),

    html.Label("Select a Gender Filter:"),
    dcc.Dropdown(
        id='gender-filter',
        options=[
            {'label': 'All', 'value': 'all'},
            {'label': 'Male', 'value': 0},
            {'label': 'Female', 'value': 1},
            # {'label': 'Trans', 'value': 2},  # Uncomment if 'Trans' exists in your data
        ],
        value='all',
        clearable=False
    ),

    dcc.Graph(id='age-dist'),
    dcc.Graph(id='time-vs-distraction'),
    dcc.Graph(id='addiction-score-dist'),
    dcc.Graph(id='scatter-addiction-time'),
])

# Callback function
@app.callback(
    [
        Output('age-dist', 'figure'),
        Output('time-vs-distraction', 'figure'),
        Output('addiction-score-dist', 'figure'),
        Output('scatter-addiction-time', 'figure')
    ],
    [Input('gender-filter', 'value')]
)
def update_graphs(selected_gender):
    if selected_gender == 'all':
        filtered_data = data
    elif selected_gender in data['Gender'].unique():
        filtered_data = data[data['Gender'] == selected_gender]
    else:
        filtered_data = pd.DataFrame(columns=data.columns) # Empty DataFrame if gender doesn't exist

    fig1 = px.histogram(filtered_data, x='What is your age?', nbins=10, title='Age Distribution', color_discrete_sequence=['royalblue'])

    fig2 = px.box(filtered_data.dropna(subset=['Time Spent on Social Media (Hours)', 'Distraction Level']),
                  x='Time Spent on Social Media (Hours)', y='Distraction Level',
                  title='Time Spent vs Distraction Level', color='Time Spent on Social Media (Hours)')

    fig3 = px.histogram(filtered_data, x='Addiction Score', nbins=20, title='Addiction Score Distribution', color_discrete_sequence=['darkred'])
    fig4 = px.scatter(filtered_data.dropna(subset=['Time Spent on Social Media (Hours)', 'Addiction Score']),
                     x='Time Spent on Social Media (Hours)', y='Addiction Score', title='Addiction Score vs Time Spent', color='Addiction Score')

    return fig1, fig2, fig3, fig4

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[316]:


get_ipython().system('pip install flask joblib scikit-learn numpy pandas')


# In[294]:


get_ipython().run_cell_magic('writefile', 'app.py', 'from flask import Flask, request, jsonify\nimport pickle\nimport numpy as np\n\n# Load the saved model\nwith open("model.pkl", "rb") as f:\n    model = pickle.load(f)\n\n# Initialize Flask app\napp = Flask(__name__)\n\n@app.route("/")\ndef home():\n    return "Welcome to the Machine Learning Model API!"\n\n@app.route("/predict", methods=["POST"])\ndef predict():\n    try:\n        # Get JSON data from the request\n        data = request.get_json()\n        \n        # Convert data into numpy array\n        features = np.array(data["features"]).reshape(1, -1)\n        \n        # Make prediction\n        prediction = model.predict(features)\n\n        # Return the result as JSON\n        return jsonify({"prediction": int(prediction[0])})\n\n    except Exception as e:\n        return jsonify({"error": str(e)})\n\n# Run the app\nif __name__ == "__main__":\n    app.run(debug=True, port=5000)\n')


# In[295]:


get_ipython().system('python app.py')


# In[296]:


import requests
import json

# API URL
url = "http://127.0.0.1:5000/predict"

# Define categorical encoding
gender_mapping = {"Male": 0, "Female": 1}
relationship_mapping = {"Single": 0, "In a relationship": 1}
time_spent_mapping = {
    "Between 2 and 3 hours": 2,
    "More than 5 hours": 5,
    "Between 3 and 4 hours": 3
}

# Sample raw data (First Row)
raw_data = {
    "age": 21,
    "gender": "Male",
    "relationship_status": "In a relationship",
    "occupation_status": "University Student",
    "social_media_usage_time": "Between 2 and 3 hours",
    "social_media_distraction": 5,
    "social_media_purpose": 3,
    "restlessness": 2,
    "distraction_scale": 5,
    "bothered_by_worries": 2,
    "concentration_difficulty": 5,
    "comparison_frequency": 2,
    "comparison_feelings": 3,
    "validation_seeking": 2,
    "depression_feeling": 5,
    "interest_fluctuation": 4,
    "sleep_issues": 5
}

# Encode categorical variables
encoded_data = {
    "features": [
        raw_data["age"],
        gender_mapping[raw_data["gender"]],
        relationship_mapping[raw_data["relationship_status"]],
        0,  # Occupation Status (All "University Student" → 0)
        time_spent_mapping[raw_data["social_media_usage_time"]],
        raw_data["social_media_distraction"],
        raw_data["social_media_purpose"],
        raw_data["restlessness"],
        raw_data["distraction_scale"],
        raw_data["bothered_by_worries"],
        raw_data["concentration_difficulty"],
        raw_data["comparison_frequency"],
        raw_data["comparison_feelings"],
        raw_data["validation_seeking"],
        raw_data["depression_feeling"],
        raw_data["interest_fluctuation"],
        raw_data["sleep_issues"]
    ]
}

# Convert to JSON
json_data = json.dumps(encoded_data)

# Send a POST request
response = requests.post(url, json=encoded_data)

# Print response
print(response.json())


# In[ ]:




