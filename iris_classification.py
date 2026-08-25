"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit

Goal: Build a basic classification model using the Iris dataset.
Pipeline (IPO Framework):
  INPUT   -> Load Iris dataset, scale features
  PROCESS -> Train-test split, KNN algorithm
  OUTPUT  -> Confusion matrix, F1 score, accuracy

Requirements:
    pip install scikit-learn pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

# -----------------------------------------------------------------
# STEP 1: INPUT — Load and understand the dataset
# -----------------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print("Dataset shape:", X.shape)
print("Classes:", dict(zip(range(3), iris.target_names)))
print("\nFirst 5 rows:\n", X.head())
print("\nClass distribution:\n", y.value_counts())

# -----------------------------------------------------------------
# STEP 2: PROCESS — Split data into training and testing sets
# -----------------------------------------------------------------
# Shuffle happens automatically inside train_test_split (random_state
# fixes the shuffle so results are reproducible).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# -----------------------------------------------------------------
# STEP 3: PROCESS — Feature scaling (the "Gatekeeper Rule")
# -----------------------------------------------------------------
# Fit the scaler ONLY on training data, then apply it to both sets
# to avoid leaking test data information into training.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# STEP 4: PROCESS — Apply KNN classification algorithm
# -----------------------------------------------------------------
# Choosing K: try a few odd values to see which gives lowest error
# (the "elbow" from the slides). Here we use K=5 as a solid default.
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# -----------------------------------------------------------------
# STEP 5: OUTPUT — Predict and validate
# -----------------------------------------------------------------
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
cm = confusion_matrix(y_test, predictions)

print(f"\n=== Results (K={k}) ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, predictions, target_names=iris.target_names))

# -----------------------------------------------------------------
# STEP 6: OUTPUT — Visualize the confusion matrix
# -----------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names, yticklabels=iris.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (KNN, K={k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion matrix plot to confusion_matrix.png")
plt.show()

# -----------------------------------------------------------------
# BONUS: Finding the optimal K (the "elbow" method from the slides)
# -----------------------------------------------------------------
error_rates = []
k_values = range(1, 21)
for k_val in k_values:
    knn = KNeighborsClassifier(n_neighbors=k_val)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, pred_k))

plt.figure(figsize=(8, 5))
plt.plot(k_values, error_rates, marker="o")
plt.xlabel("K value")
plt.ylabel("Error Rate")
plt.title("Choosing the Optimal K")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig("k_tuning.png", dpi=150)
print("Saved K-tuning plot to k_tuning.png")
plt.show()
