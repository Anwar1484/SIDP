import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CSV_FILE = "posture_dataset.csv"
MODEL_FILE = "trained_posture_model.pkl"

if not os.path.exists(CSV_FILE):
    print(f"[ERROR] {CSV_FILE} not found! Please run 1_collect_data.py first.")
    exit()

# 1. Load the collected CSV dataset, drop incomplete rows, and remove 'walking'
df = pd.read_csv(CSV_FILE).dropna()
walking_count = int((df["label"] == "walking").sum())
df = df[df["label"] != "walking"].reset_index(drop=True)
df.to_csv(CSV_FILE, index=False)

print("=======================================================")
print("   STEP 2: TRAINING THE 4-CLASS AI POSTURE MODEL")
print("=======================================================")
if walking_count > 0:
    print(f"[CLEANED] Removed {walking_count} 'walking' rows and updated {CSV_FILE}.")
print(f"Total 4-class samples collected: {len(df)}\n")
print("Samples per posture:")
print(df["label"].value_counts())
print("-------------------------------------------------------")

# Separate the features (68 numbers) and the target label (posture name)
# Convert explicitly to pure NumPy arrays to bypass PyArrow string errors
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y = np.array(df["label"].tolist(), dtype=str)

# 2. Split data: 80% for teaching the AI, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create and train the AI Model (Random Forest Classifier)
print("Training AI model now...")
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 4. Test the AI Model on the 20% unseen test data
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"\n---> OVERALL AI ACCURACY: {acc:.2f}% <---\n")
print("Detailed Accuracy Report (Screenshot this for your Supervisor!):")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Save the trained AI brain to a file
joblib.dump(model, MODEL_FILE)
print("=======================================================")
print(f"[SUCCESS] Trained AI model saved as: {MODEL_FILE}")
print("=======================================================")