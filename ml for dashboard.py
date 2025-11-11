import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Step 1: Load both datasets ---
dfMarket = pd.read_csv("market_pipe_thickness_loss_dataset.csv")   # Market dataset

# Basic info
print("Dataset 1 info:")
dfMarket.info()

# Summary statistics
print("\nDataset 1 describe:")
print(dfMarket.describe())

print("Dataset shape: ", dfMarket.shape)

#train and test (alr clean, process, encode, feature engineering, feature selection)
safe_featuresFS20 = ['loss_ratio',  'pipe_size_mm_tf','corrosion_impact_percent', 'thickness_mm','normalized_thickness',
                     'material_risk_index', 'material_fiberglass','age_group_1–5 years','pressure_stress_interaction','material_carbon steel',
                      'Stress_Index','size_pressure_index','pressure_to_thickness','pressure_to_size','thickness_decay_ratio',
                     'pressure_temp_interaction','material_pvc', 'Operational_Stress_Age','age_group_6–10 years','max_pressure_psi']

input_data = dfMarket[safe_featuresFS20]
output_data = dfMarket["condition_encoded"]

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(
    input_data,
    output_data)

x_train, x_test, y_train, y_test = train_test_split(
    x_resampled,
    y_resampled,
    test_size=0.25,
    stratify=y_resampled,
    random_state=42
)
# ===== TUNED XGBOOST MODEL =====

# Parameter grid
params_xgb = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

# Base model
xgb_base = XGBClassifier(
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='mlogloss'
)

# GridSearchCV
grid_xgb = GridSearchCV(
    xgb_base,
    params_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit model
grid_xgb.fit(x_train, y_train)
best_xgb_model = grid_xgb.best_estimator_

# Predictions
y_pred_xgb = best_xgb_model.predict(x_test)

# ===== ENSEMBLE MODELS COMPARISON - Voting (xgb + lgbm + rf) =====
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Reuse tuned XGBoost
xgb_best = best_xgb_model

# Common function to evaluate any ensemble
def evaluate_ensemble(model, name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Cross-validation
    cv_acc = cross_val_score(model, input_data, output_data, cv=5, scoring='accuracy')
    cv_f1 = cross_val_score(model, input_data, output_data, cv=5, scoring='f1_macro')

    results = {
        "Model Name": name,
        "Best XGB Params": grid_xgb.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Macro": f1_score(y_test, y_pred, average='macro'),
        "Report": classification_report(y_test, y_pred, target_names=['Normal', 'Moderate', 'Critical']),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "CV Accuracy": cv_acc,
        "Mean CV Accuracy": cv_acc.mean(),
        "CV F1 Macro": cv_f1,
        "Mean CV F1 Macro": cv_f1.mean()
    }

    print(f"\n=== {name} RESULTS ===")
    print("Best XGB Parameters:", results["Best XGB Params"])
    print(f"Accuracy: {results['Accuracy']:.3f}")
    print(f"F1 Macro: {results['F1 Macro']:.3f}")
    print("\nClassification Report:\n", results["Report"])
    print("Confusion Matrix:\n", results["Confusion Matrix"])
    print("\n=== CROSS VALIDATION ===")
    print("CV Accuracy:", results["CV Accuracy"])
    print("Mean CV Accuracy:", results["Mean CV Accuracy"])
    print("CV F1 Macro:", results["CV F1 Macro"])
    print("Mean CV F1 Macro:", results["Mean CV F1 Macro"])

    return results
# ===== SET 2: Tree Powerhouse (XGB + LGBM + Random Forest) =====
lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42
)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

voting_clf2 = VotingClassifier(
    estimators=[('xgb', xgb_best), ('lgbm', lgbm_model), ('rf', rf_model)],
    voting='soft'
)
results_set2 = evaluate_ensemble(voting_clf2, "Voting Ensemble - XGB + LGBM + RF")

# ===== ADD PREDICTIONS TO ORIGINAL DATASET & EXPORT =====

# Use your best model - Voting Ensemble (XGB + LGBM + RF)
production_model = voting_clf2

# Get predictions for all data
print("Generating predictions for all pipelines...")
all_predictions = production_model.predict(input_data)
all_probabilities = production_model.predict_proba(input_data)

# Add predictions directly to original dfMarket
dfMarket['Predicted_Condition'] = all_predictions
dfMarket['Predicted_Condition_Label'] = [
    'Normal' if x == 0 else 'Moderate' if x == 1 else 'Critical'
    for x in all_predictions
]

# Add probability scores
dfMarket['Probability_Normal'] = all_probabilities[:, 0]
dfMarket['Probability_Moderate'] = all_probabilities[:, 1]
dfMarket['Probability_Critical'] = all_probabilities[:, 2]

# Add confidence score (highest probability)
dfMarket['Prediction_Confidence'] = all_probabilities.max(axis=1)

# Add simple risk level for dashboard
dfMarket['Risk_Level'] = dfMarket['Predicted_Condition_Label'].map({
    'Normal': 'Low Risk',
    'Moderate': 'Medium Risk',
    'Critical': 'High Risk'
})

# Export to CSV
output_filename = 'pipeline_data_with_predictions.csv'
dfMarket.to_csv(output_filename, index=False)

print("✅ PREDICTIONS ADDED AND EXPORTED!")
print(f"📁 File saved: {output_filename}")
print(f"📊 Total pipelines: {len(dfMarket)}")
print(f"🎯 Model used: Voting Ensemble (XGB + LGBM + RF)")
print(f"📈 Model accuracy: {results_set2['Accuracy']:.3f}")

print(f"\n🔍 PREDICTION DISTRIBUTION:")
prediction_counts = dfMarket['Predicted_Condition_Label'].value_counts()
for condition, count in prediction_counts.items():
    percentage = (count / len(dfMarket)) * 100
    print(f"   {condition}: {count} pipelines ({percentage:.1f}%)")

print(f"\n📋 NEW COLUMNS ADDED:")
new_columns = ['Predicted_Condition', 'Predicted_Condition_Label',
               'Probability_Normal', 'Probability_Moderate', 'Probability_Critical',
               'Prediction_Confidence', 'Risk_Level']
for col in new_columns:
    print(f"   ✓ {col}")

print(f"\n📊 SAMPLE OF FINAL DATASET:")
sample_preview = dfMarket[['thickness_mm', 'corrosion_impact_percent',
                          'Predicted_Condition_Label', 'Risk_Level',
                          'Prediction_Confidence']].head(5)
print(sample_preview)

# Add Prediction_Correct and Prediction_Confidence columns to dfMarket
dfMarket['Prediction_Correct'] = dfMarket['condition_encoded'] == dfMarket['Predicted_Condition']
dfMarket['Prediction_Confidence'] = dfMarket[['Probability_Normal', 'Probability_Moderate', 'Probability_Critical']].max(axis=1)

# Add Actual_Condition_Label for completeness
dfMarket['Actual_Condition_Label'] = dfMarket['condition_encoded'].map({
    0: 'Normal',
    1: 'Moderate',
    2: 'Critical'
})

import joblib

# Save the trained voting ensemble
joblib.dump(production_model, "corrosion_voting_model.pkl")


print("Hello ")