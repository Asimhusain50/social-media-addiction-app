#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("/Users/asim/Downloads/Students Social Media Addiction.csv")



# In[3]:


df


# In[7]:


# Basic Info
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Countplots for Categorical
print(df['Gender'].value_counts())
print(df['Most_Used_Platform'].value_counts())
print(df['Affects_Academic_Performance'].value_counts())

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Histogram of Daily Usage
sns.histplot(df['Avg_Daily_Usage_Hours'], kde=True, bins=20)
plt.title("Avg Daily Usage")
plt.show()

# Boxplot - Addiction by Gender
sns.boxplot(x='Gender', y='Addicted_Score', data=df)
plt.title("Addiction Score by Gender")
plt.show()

# Most Used Platform
plt.figure(figsize=(10, 5))
sns.countplot(y='Most_Used_Platform', data=df, order=df['Most_Used_Platform'].value_counts().index)
plt.title("Most Used Platforms")
plt.show()

# Scatterplot - Usage vs Addiction
sns.scatterplot(x='Avg_Daily_Usage_Hours', y='Addicted_Score', hue='Gender', data=df)
plt.title("Usage Hours vs Addicted Score")
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

# Create binary target: Addicted (1 if score >= 7)
df['Addicted'] = df['Addicted_Score'].apply(lambda x: 1 if x >= 7 else 0)

# Drop unnecessary columns
X = df.drop(columns=['Student_ID', 'Addicted_Score', 'Addicted'])

# Encode categorical variables
X_encoded = pd.get_dummies(X)

y = df['Addicted']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[9]:


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


# In[10]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Accuracy and Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[11]:


importances = rf.feature_importances_
features = X_encoded.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feat_df.head(10))
plt.title("Top 10 Important Features")
plt.show()


# In[12]:


from sklearn.metrics import roc_curve, auc

y_prob = rf.predict_proba(X_test)[:,1]  # Get probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[13]:


import joblib
from sklearn.ensemble import RandomForestClassifier

# Train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Save the model and columns
joblib.dump(rf, 'model.pkl')
joblib.dump(X_encoded.columns.tolist(), 'columns.pkl')


# In[ ]:





# In[15]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Drop identifiers and labels
unsup_df = df.drop(columns=['Student_ID', 'Addicted_Score', 'Addicted'])

# One-hot encode categorical features
unsup_encoded = pd.get_dummies(unsup_df)

# Standardize features (important for clustering)
scaler = StandardScaler()
unsup_scaled = scaler.fit_transform(unsup_encoded)


# In[21]:


from sklearn.cluster import KMeans

# Try 3 clusters for example
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(unsup_scaled)

# Add cluster label to original dataframe
df['Cluster'] = kmeans.labels_


# In[22]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
reduced = pca.fit_transform(unsup_scaled)

df['PCA1'] = reduced[:,0]
df['PCA2'] = reduced[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Student Clusters (PCA reduced)')
plt.show()


# In[25]:


# Look at mean values per cluster
cluster_profile = df.groupby('Cluster').mean(numeric_only=True)
print(cluster_profile)


# In[27]:


inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(unsup_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# In[30]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse encoded and scaled data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pd.get_dummies(df.drop(columns=['Student_ID', 'Addicted_Score', 'Addicted'])))

# DBSCAN model
dbscan = DBSCAN(eps=1.5, min_samples=5)
db_labels = dbscan.fit_predict(scaled_data)

# Add DBSCAN cluster labels to original DataFrame
df['DBSCAN_Cluster'] = db_labels

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PCA1'] = pca_data[:,0]
df['PCA2'] = pca_data[:,1]

# Plot clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='DBSCAN_Cluster', palette='tab10')
plt.title('DBSCAN Clustering (PCA Reduced)')
plt.show()


# In[33]:


from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Use linkage method on scaled data
linked = linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()


# In[35]:


import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your dataset (ensure it's preloaded)
df['Addicted'] = df['Addicted_Score'].apply(lambda x: 1 if x >= 7 else 0)
X = df.drop(columns=['Student_ID', 'Addicted_Score', 'Addicted'])
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Define clustering function
def cluster_and_plot(method, n_clusters=3, eps=1.5, min_samples=5):
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif method == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

    # Reduce dimensions with PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", legend="full")
    plt.title(f"{method} Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()

    # Save figure
    plot_path = "cluster_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 🧠 Student Social Media Cluster Explorer")

    with gr.Row():
        method = gr.Radio(["KMeans", "DBSCAN"], label="Choose Clustering Method", value="KMeans")
        k_slider = gr.Slider(2, 10, step=1, label="K (for KMeans)", value=3)
        eps_slider = gr.Slider(0.1, 3.0, step=0.1, label="Epsilon (for DBSCAN)", value=1.5)
        min_samples_slider = gr.Slider(2, 20, step=1, label="Min Samples (for DBSCAN)", value=5)

    btn = gr.Button("Run Clustering")

    output = gr.Image(label="Cluster Plot")

    def run(method, n_clusters, eps, min_samples):
        return cluster_and_plot(method, n_clusters, eps, min_samples)

    btn.click(fn=run, inputs=[method, k_slider, eps_slider, min_samples_slider], outputs=output)

# Launch the app
app.launch()


# In[39]:


from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso.fit_predict(X_scaled)

# Add result to DataFrame
df['Anomaly'] = anomaly_labels

# Visualize anomalies in PCA space
df['Anomaly'] = anomaly_labels

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Anomaly', palette={1: 'blue', -1: 'red'})
plt.title('Isolation Forest Anomaly Detection')
plt.show()


# In[41]:


from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(X_scaled)

df['LOF_Anomaly'] = lof_labels
df['LOF_Anomaly'] = lof_labels

# Plot LOF result
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='LOF_Anomaly', palette={1: 'green', -1: 'orange'})
plt.title('LOF Anomaly Detection')
plt.show()


# In[43]:


import gradio as gr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load data

df['Addicted'] = df['Addicted_Score'].apply(lambda x: 1 if x >= 7 else 0)
X = df.drop(columns=['Student_ID', 'Addicted_Score', 'Addicted'])
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, df['Addicted'])
joblib.dump(model, "model.pkl")
joblib.dump(X_encoded.columns.tolist(), "columns.pkl")

# Plot to image function
def plot_to_image(plt_func):
    plt_func()
    filename = "plot.png"
    plt.savefig(filename)
    plt.close()
    return filename

# EDA Plot
def eda_plot():
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    return plot_to_image(plt.show)

# Supervised Prediction
def predict_addiction(
    Age, Gender, Academic_Level, Country, Avg_Daily_Usage_Hours,
    Most_Used_Platform, Affects_Academic_Performance, Sleep_Hours_Per_Night,
    Mental_Health_Score, Relationship_Status, Conflicts_Over_Social_Media
):
    input_dict = {
        "Age": Age,
        "Gender": Gender,
        "Academic_Level": Academic_Level,
        "Country": Country,
        "Avg_Daily_Usage_Hours": Avg_Daily_Usage_Hours,
        "Most_Used_Platform": Most_Used_Platform,
        "Affects_Academic_Performance": Affects_Academic_Performance,
        "Sleep_Hours_Per_Night": Sleep_Hours_Per_Night,
        "Mental_Health_Score": Mental_Health_Score,
        "Relationship_Status": Relationship_Status,
        "Conflicts_Over_Social_Media": Conflicts_Over_Social_Media
    }

    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df).reindex(columns=columns, fill_value=0)
    pred = model.predict(input_df)[0]
    return "Addicted" if pred == 1 else "Not Addicted"


# Unsupervised Clustering
def cluster_plot(method, k, eps, min_samples):
    if method == "KMeans":
        model = KMeans(n_clusters=k)
        labels = model.fit_predict(X_scaled)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    df_temp = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
    df_temp['Cluster'] = labels

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_temp, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
    plt.title(f"{method} Clustering")
    return plot_to_image(plt.show)

# PCA Projection
def pca_projection():
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    df_temp = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
    df_temp['Addicted'] = df['Addicted']

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_temp, x='PCA1', y='PCA2', hue='Addicted', palette='Set1')
    plt.title("PCA Projection of Addicted vs Not")
    return plot_to_image(plt.show)

# Anomaly Detection
def anomaly_plot(method):
    if method == "Isolation Forest":
        iso = IsolationForest(contamination=0.05)
        labels = iso.fit_predict(X_scaled)
    else:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        labels = lof.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    df_temp = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
    df_temp['Anomaly'] = labels

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_temp, x='PCA1', y='PCA2', hue='Anomaly', palette={1: 'blue', -1: 'red'})
    plt.title(f"Anomaly Detection using {method}")
    return plot_to_image(plt.show)

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📊 Social Media Addiction Analysis App")

    with gr.Tab("1️⃣ EDA"):
        eda_btn = gr.Button("Show Correlation Heatmap")
        eda_out = gr.Image()
        eda_btn.click(eda_plot, outputs=eda_out)

    with gr.Tab("2️⃣ Predict Addiction (Supervised)"):
        with gr.Row():
            inputs = [
                gr.Number(label="Age"),
                gr.Dropdown(["Male", "Female"], label="Gender"),
                gr.Textbox(label="Academic_Level"),
                gr.Textbox(label="Country"),
                gr.Number(label="Avg_Daily_Usage_Hours"),
                gr.Textbox(label="Most_Used_Platform"),
                gr.Radio(["Yes", "No"], label="Affects_Academic_Performance"),
                gr.Number(label="Sleep_Hours_Per_Night"),
                gr.Number(label="Mental_Health_Score"),
                gr.Textbox(label="Relationship_Status"),
                gr.Number(label="Conflicts_Over_Social_Media")
            ]
        pred_btn = gr.Button("Predict")
        pred_out = gr.Textbox()
        pred_btn.click(predict_addiction, inputs=inputs, outputs=pred_out)

    with gr.Tab("3️⃣ Clustering (Unsupervised)"):
        method = gr.Radio(["KMeans", "DBSCAN"], label="Method")
        k = gr.Slider(2, 10, step=1, label="K (for KMeans)", value=3)
        eps = gr.Slider(0.1, 3.0, step=0.1, label="Epsilon (for DBSCAN)", value=1.5)
        min_samples = gr.Slider(2, 20, step=1, label="Min Samples (DBSCAN)", value=5)
        cluster_btn = gr.Button("Run Clustering")
        cluster_out = gr.Image()
        cluster_btn.click(cluster_plot, inputs=[method, k, eps, min_samples], outputs=cluster_out)

    with gr.Tab("4️⃣ PCA Projection"):
        pca_btn = gr.Button("Project 2D PCA View")
        pca_out = gr.Image()
        pca_btn.click(pca_projection, outputs=pca_out)

    with gr.Tab("5️⃣ Anomaly Detection"):
        method = gr.Radio(["Isolation Forest", "Local Outlier Factor"], label="Anomaly Detection Method")
        anom_btn = gr.Button("Detect Outliers")
        anom_out = gr.Image()
        anom_btn.click(anomaly_plot, inputs=[method], outputs=anom_out)

demo.launch(share=True)


# In[45]:


df


# In[128]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize and train
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict
y_pred_log = log_model.predict(X_test)

# Evaluate
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))


# In[130]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Predict (continuous)
y_pred_lin = lin_model.predict(X_test)

# Round to 0 or 1 (optional)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred_lin]

# Evaluate
print("Linear Regression Results (rounded):")
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

# Also check MSE and R² if treating it as regression
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R² Score:", r2_score(y_test, y_pred_lin))


# In[ ]:





# In[ ]:import os
import pickle

# Correct way to load model file on Streamlit Cloud
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)





