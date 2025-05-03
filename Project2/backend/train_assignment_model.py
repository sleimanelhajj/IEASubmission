import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.multioutput import MultiOutputClassifier
with open("ga_training_data.json") as f:
    data = json.load(f)

X = []
y = []
for sample in data:
    features = np.array(sample["grid"]).flatten().tolist()
    for agent in sample["agents"]:
        features += list(agent)
    for target in sample["targets"]:
        features += list(target)
    for obstacle in sample["obstacles"]:
        features += list(obstacle)
    X.append(features)
    # Use the full assignment list as the label
    y.append(sample["assignment"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multi-output classifier
clf = MultiOutputClassifier(RandomForestClassifier())
clf.fit(X_train, y_train)
print("Model trained!")

# Evaluate accuracy (per agent)
y_pred = clf.predict(X_test)
y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_pred)
mean_agent_accuracy = (y_test_arr == y_pred_arr).mean()
print(f"Mean assignment accuracy (all agents): {mean_agent_accuracy:.3f}")

# Save model
joblib.dump(clf, "assignment_model_multi.pkl")