import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------- LOAD YOUR DATA ----------
df = pd.read_csv("outputs/all_studies_noisy.csv")

# ---------- SAFELY CREATE Total_Issues ----------
if "Total_Issues" not in df.columns:
    df["Total_Issues"] = (
        df["SAE_Pending_Count"]
        + df["Overdue_Visits_Count"]
        + df["Missing_Pages"]
    )

# ---------- PREPARE TARGET ----------
df["TargetHigh"] = (df["Smart_Risk_Noisy"] == "High").astype(int)

# ---------- FEATURES ----------
feature_cols = [
    "Smart_DQI_Noisy",
    "SAE_Pending_Count",
    "Overdue_Visits_Count",
    "Missing_Pages",
    "Total_Issues"
]

# ensure every feature exists
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0

X = df[feature_cols]
y = df["TargetHigh"]

# ---------- TRAIN MODEL ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------- SAVE MODEL ----------
joblib.dump((model, feature_cols), "early_warning_model.pkl")

print("✅ Model training complete.")
print("💾 Saved as early_warning_model.pkl")
