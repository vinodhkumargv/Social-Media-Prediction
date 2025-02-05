#!/usr/bin/env python
# coding: utf-8

# ## Data Loading and Cleaning 

# In[201]:


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
print(data.isnull().sum())


# Display the first few rows
data.head()


print(data['Gender'].unique())




# ## Data Cleaning

# In[202]:


# Handle missing values (if any)
data = data.dropna()  # or use data.fillna() for imputation

# Convert categorical variables to numerical (if needed)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1, 'Trans': 2})  # -1 often represents 'Other' or 'Unknown'
data['Relationship Status'] = data['Relationship Status'].map({'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3})


# Check the cleaned data
data.head()


# ## EDA

# In[203]:


plt.figure(figsize=(10, 6))  # Increase figure size
sns.histplot(data['What is your age?'], bins=10, kde=True, color="royalblue", edgecolor="black", alpha=0.7)  # Add transparency
sns.kdeplot(data['What is your age?'], color="darkred", linewidth=2)  # Customize KDE line

# Titles and labels
plt.title("Age Distribution", fontsize=14, fontweight="bold", color="darkblue")
plt.xlabel("Age (Years)", fontsize=12, fontweight="bold")
plt.ylabel("Frequency", fontsize=12, fontweight="bold")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle="--", alpha=0.6)  # Add dashed gridlines

# Show the chart
plt.show()


#Distibution of Time Spent on Social Media

plt.figure(figsize=(10, 6))
sns.histplot(data['What is the average time you spend on social media every day?'], bins=10, kde=True, color="purple")
plt.title("Distribution of Time Spent on Social Media", fontsize=14)
plt.xlabel("Time Spent (Hours per Day)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# In[204]:


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


# In[205]:


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





# In[206]:


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

data.head()

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


# In[207]:


# EDA: Distribution of Addiction Score
plt.figure(figsize=(8, 6))
sns.histplot(data['Addiction Score'], kde=True, bins=20, color='blue')
plt.title("Distribution of Addiction Score")
plt.xlabel("Addiction Score")
plt.ylabel("Frequency")
plt.show()


# ## Predictive Analysis

# In[ ]:





# In[208]:


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


# ## Model Training

# In[213]:


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

# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




# In[214]:


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


# In[215]:


# Feature Importance
plt.figure(figsize=(12, 6))
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# In[220]:


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


# In[221]:


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


# In[225]:


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




# In[231]:


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


# In[232]:


import joblib

# Save the trained model to a file
joblib.dump(model, 'random_forest_classifier.pkl')


# In[235]:


from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the POST request
    data = request.get_json(force=True)
    
    # Convert the JSON data to a DataFrame
    input_data = pd.DataFrame(data, index=[0])
    
    # Make predictions using the loaded model
    predictions = model.predict(input_data)
    
    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)


# In[236]:


get_ipython().system('python app.py')


# In[237]:


with open("app.py", "w") as file:
    file.write("""
    from flask import Flask, request, jsonify
    import joblib
    import pandas as pd

    app = Flask(__name__)

    # Load the trained model
    model = joblib.load('random_forest_classifier.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        # Get the JSON data from the POST request
        data = request.get_json(force=True)
        
        # Convert the JSON data to a DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Make predictions using the loaded model
        predictions = model.predict(input_data)
        
        # Return the predictions as a JSON response
        return jsonify(predictions.tolist())

    if __name__ == '__main__':
        app.run(debug=True)
    """)


# In[ ]:





# In[ ]:




