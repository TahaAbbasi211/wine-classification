# ==========================
# Wine Classification Project
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def load_data():
    """Load the Wine dataset and return features, target, and class names."""
    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    print("Dataset shape:", X.shape)
    print("Target classes:", list(wine.target_names))
    print("Missing values:", int(X.isnull().sum().sum()))

    return X, y, wine.target_names

# ==========================
#  Explore Data
# ==========================

def explore_data(X, y):
    """Print basic information about the dataset."""
    df = X.copy()
    df["target"] = y

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    return df

# ==========================
#  Build Models
# ==========================

def build_models():
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000)
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

    return models

# ==========================
#  Visualize Data
# ==========================

def visualize_data(df, X):
    """Create exploratory plots."""
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="target")
    plt.title("Number of Samples in Each Class")
    plt.xlabel("Wine Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="alcohol",
        y="color_intensity",
        hue="target",
    )
    plt.title("Alcohol vs Color Intensity")
    plt.xlabel("Alcohol")
    plt.ylabel("Color Intensity")
    plt.tight_layout()
    plt.show()

    correlation = X.corr()
    print("\nFeature correlations:")
    print(correlation.round(2))

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def prepare_data(X, y):
    """Split the data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test


def build_models():
    """Create the models that will be compared."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

# ==========================
# 5. Evaluate Models
# ==========================

def evaluate_models(models, X_train, X_test, y_train, y_test, target_names):
    """Train and compare models, then report the top model's test results."""
    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        results.append({"Model": name, "Accuracy": accuracy})
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy",
        ascending=False,
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Accuracy", y="Model")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

    # For this learning project, report the model with the highest test accuracy.
    # In a more rigorous workflow, use cross-validation for model selection.
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    print(f"\nBest model by test accuracy: {best_name}")
    print("Accuracy:", round(accuracy_score(y_test, best_predictions), 4))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            best_predictions,
            target_names=target_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.show()

    return results_df


def main():
    """Run the complete Wine Classification workflow."""
    X, y, target_names = load_data()
    df = explore_data(X, y)
    visualize_data(df, X)

    X_train, X_test, y_train, y_test = prepare_data(X, y)
    models = build_models()

    evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
    )


if __name__ == "__main__":
    main()
