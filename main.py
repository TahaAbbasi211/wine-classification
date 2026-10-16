# ==========================
# 1. Import Libraries
# ==========================

from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 2. Load Dataset
# ==========================
wine = load_wine()

# Convert features into a DataFrame
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Add the target column
df["target"] = wine.target

# Inspect the dataset
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class names:", wine.target_names)

# ==========================
# 3. Explore Data
# ==========================

print(df.head())
print(df.info())
print(df.describe())

# ==========================
# 4. Visualize Data
# ==========================

sns.countplot(data=df, x="target")

plt.title("Number of Samples in Each Class")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# ==========================
# 5. Feature Relationships
# ==========================

sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="target"
)

plt.title("Alcohol vs Color Intensity")
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.tight_layout()
plt.show()

# ==========================
# 6. Correlation Analysis
# ==========================

features = df.drop(columns="target")

correlation = features.corr()

print(correlation.round(2))

# ==========================
# 7. Correlation Heatmap
# ==========================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ==========================
# 8. Prepare Features and Target
# ==========================

X = df.drop(columns="target")
y = df["target"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Number of features:", X.shape[1])
print("Number of classes:", y.nunique())

# ==========================
# 9. Train/Test Split
# ==========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================
# 10. Train the Model
# ==========================

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)

model.fit(X_train, y_train)

print("Model training completed!")

# ==========================
# 11. Make Predictions
# ==========================

y_pred = model.predict(X_test)

print("Actual classes:    ", y_test.iloc[:10].tolist())
print("Predicted classes: ", y_pred[:10].tolist())

# ==========================
# 12. Evaluate the Model
# ==========================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("Accuracy percentage:", round(accuracy * 100, 2), "%")

# ==========================
# 13. Compare Models
# ==========================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print(results_df.to_string(index=False))

# ==========================
# 14. Visualize Model Comparison
# ==========================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.xlim(0, 1)

plt.tight_layout()
plt.show()