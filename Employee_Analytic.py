import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# file_path = "C:\\Users\\KIIT\\Desktop\\ADL\\india_job_market_dataset.csv"
file_path = "C:\\Users\\KIIT\\Desktop\\ADL\\india_job_market_dataset.dat"
df = pd.read_csv(file_path)

# Data Preprocessing
# Convert Salary Range into numerical values
def parse_salary(salary):
    if '+' in salary:
        return float(salary.replace('+ LPA', ''))
    elif '-' in salary:
        low, high = salary.replace(' LPA', '').split('-')
        return (float(low) + float(high)) / 2
    return np.nan

df["Salary (LPA)"] = df["Salary Range"].apply(parse_salary)

# Convert Experience Required into numerical values
def parse_experience(exp):
    if '10+' in exp:
        return 10
    elif '-' in exp:
        low, high = exp.replace(' years', '').split('-')
        return (int(low) + int(high)) / 2
    return np.nan

df["Experience (Years)"] = df["Experience Required"].apply(parse_experience)

# Encode categorical variables
df["Remote/Onsite"] = df["Remote/Onsite"].map({"Remote": 0, "Hybrid": 1, "Onsite": 2})
df["Job Type"] = df["Job Type"].astype("category").cat.codes
df["Company Size"] = df["Company Size"].astype("category").cat.codes

# Drop unused columns
df_cleaned = df.drop(columns=["Job ID", "Posted Date", "Application Deadline", "Job Portal", "Salary Range", "Experience Required"])

# Salary Prediction Model (Random Forest Regression)
features = ["Job Type", "Number of Applicants", "Remote/Onsite", "Company Size", "Experience (Years)"]
target = "Salary (LPA)"

X = df_cleaned[features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Salary Prediction RMSE: {rmse:.2f} LPA")

# Job-Candidate Matching (TF-IDF + Cosine Similarity)
job_descriptions = df_cleaned["Job Title"] + " " + df_cleaned["Skills Required"]
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(job_descriptions)
similarity_matrix = cosine_similarity(job_vectors)

def recommend_jobs(job_index, top_n=5):
    sim_scores = list(enumerate(similarity_matrix[job_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_jobs = sim_scores[1:top_n+1]
    return df_cleaned.iloc[[i[0] for i in top_jobs]][["Job Title", "Company Name", "Skills Required", "Salary (LPA)"]]

# Example Recommendation
print("Recommended Jobs:")
print(recommend_jobs(0))

# Job Market Trend Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["Salary (LPA)"], bins=20, kde=True)
plt.title("Salary Distribution in Job Market")
plt.xlabel("Salary (LPA)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=df_cleaned["Remote/Onsite"], y=df_cleaned["Salary (LPA)"])
plt.title("Salary Comparison for Remote, Hybrid, and Onsite Jobs")
plt.xlabel("Job Type (0=Remote, 1=Hybrid, 2=Onsite)")
plt.ylabel("Salary (LPA)")
plt.show()