import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np

# --- Step 1: Load both datasets ---
dfMarket = pd.read_csv("market_pipe_thickness_loss_dataset.csv")   # Market dataset

# Basic info
print("Dataset 1 info:")
dfMarket.info()

# Summary statistics
print("\nDataset 1 describe:")
print(dfMarket.describe())

print("Dataset shape: ", dfMarket.shape)

#2. DATA CLEANING
 #(handle missing values & duplicates,
 #rename column name into small letters,
 #check unique values,
 # handle outliers)

#Removing duplicates & missing values

print("\nMissing values count:")
print(dfMarket.isnull().sum())  # Check for null values
print("\nDuplicate rows count:")
print(dfMarket.duplicated().sum())  # Identify duplicate records
print("\nMarket dataset's number rows and columns: ", dfMarket.shape)

sns.heatmap(dfMarket.isnull())
plt.show()

 #rename column name into small letters,
dfMarket.columns = dfMarket.columns.str.strip().str.lower()
print(dfMarket.columns)


 #check unique values,

dfMarket['material'] = dfMarket['material'].str.lower().str.replace('-', ' ').str.strip()
dfMarket['grade'] = dfMarket['grade'].str.lower().str.replace('-', ' ').str.strip()
dfMarket['condition'] = dfMarket['condition'].str.lower().str.replace('-', ' ').str.strip()

print("Unique values for material column: ",dfMarket['material'].unique())
print("Unique values for condition column: ",dfMarket['condition'].unique())
print("Unique values for grade column: ",dfMarket['grade'].unique())

 # checking and handle outliers for dataset Market)

# Market dataset
numeric_cols_market = ['pipe_size_mm', 'thickness_mm', 'max_pressure_psi',
                       'temperature_c', 'corrosion_impact_percent',
                       'thickness_loss_mm', 'material_loss_percent', 'time_years']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols_market, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=dfMarket[col], color='lightblue')
    plt.title(col)
plt.tight_layout()
plt.show()

#BASIC DATA CLEANING
#fixing outliers

# Define a function to detect outlier bounds using IQR
def find_outlier_bounds(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# Find lower and upper bounds for the two columns (tells what is considered normal, other than that is an outlier)
lb_thick, ub_thick = find_outlier_bounds(dfMarket, 'thickness_mm')
lb_loss, ub_loss = find_outlier_bounds(dfMarket, 'material_loss_percent')

print(f"thickness_mm bounds: {lb_thick:.2f} to {ub_thick:.2f}")
print(f"material_loss_percent bounds: {lb_loss:.2f} to {ub_loss:.2f}")
  #CONC: will be only handling outliers in material_loss_percent since it can be possible/logic for thickness_mm (heavy duty pipe)

#check how many rows have values below the lower bound and above the upper bound
outliers_below = (dfMarket['material_loss_percent'] < lb_loss).sum()
outliers_above = (dfMarket['material_loss_percent'] > ub_loss).sum()
print("\n---Details of outliers in material_loss_percent---")
print(f"Outliers below lower bound: {outliers_below}")
print(f"Outliers above upper bound: {outliers_above}")

#Capping/Winsorizing = fix the outliers to nearest valid boundary
dfMarket['material_loss_percent'] = dfMarket['material_loss_percent'].clip(lb_loss, ub_loss)

#visualizing
plt.figure(figsize=(6,4))
sns.boxplot(x=dfMarket['material_loss_percent'], color='skyblue')
plt.title("Boxplot after outlier capping (material_loss_percent)")
plt.show()

#distribution plot of Material Loss Percent after capping outliers
displotMaterialLossAf = sns.displot(dfMarket['material_loss_percent'], kde=True, stat='density')
plt.show()


#dropping grade column for both dataset since it has no context
dfMarket = dfMarket.drop(columns=['grade'])

print("Current columns in dataset: ", dfMarket.columns)

#3. feature engineering on categorical data on 4 cat columns : material, condition

for col in ['material', 'condition']:
    print(f"{col} unique values:\n{dfMarket[col].unique()}\n")

# --- ONE-HOT ENCODING FOR 'material' COLUMN ---

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Select the categorical column
catData = dfMarket[['material']]

# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

# Fit and transform the data
encoded_array = ohe.fit_transform(catData)

# Convert the encoded array back into a DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(['material']))

# Combine encoded columns with original dataframe (optional)
df_encoded = pd.concat([dfMarket.reset_index(drop=True), encoded_df], axis=1)

# Display final encoded dataframe
print(df_encoded.head())

# --- LABEL ENCODING FOR 'condition' COLUMN ---

# Define the order of severity
condition_mapping = {
    'normal': 0,
    'moderate': 1,
    'critical': 2
}

# Apply the mapping to the original dataframe - use map and not Labelencoder/OrdinalEncoder coz have full control over order
dfMarket['condition_encoded'] = dfMarket['condition'].map(condition_mapping)

# Display the first few rows
print(dfMarket[['condition', 'condition_encoded']].head())
# Combine the encoded material columns with the condition-encoded column
dfMarket = pd.concat([df_encoded.reset_index(drop=True),
                      dfMarket[['condition_encoded']].reset_index(drop=True)], axis=1)

#FE on numerical data
#feature 1: Corrosion Rate (mm per year)
#Measures how fast the pipe loses thickness each year.
dfMarket['corrosion_rate_mm_per_year'] = dfMarket['thickness_loss_mm'] / dfMarket['time_years']
print(dfMarket['corrosion_rate_mm_per_year'])

#feature 2: Pressure to Thickness Ratio
#Higher ratio → higher stress on the pipe wall → faster corrosion risk.
dfMarket['pressure_to_thickness'] = dfMarket['max_pressure_psi'] / dfMarket['thickness_mm']
print(dfMarket['pressure_to_thickness'])

#feature 3: material loss ratio
#shows percentage of material loss compared to total corrosion impact
dfMarket['loss_ratio'] = dfMarket['material_loss_percent'] / (dfMarket['corrosion_impact_percent'] + 1e-6)
print(dfMarket['loss_ratio'])

#feature engineering on categorical data
# --- Material Risk Index based on corrosion susceptibility ---
risk_map = {
    'material_carbon steel': 3,       # High risk
    'material_fiberglass': 0,         # Very low risk
    'material_hdpe': 0,               # Very low risk
    'material_pvc': 0,                # Very low risk
    'material_stainless steel': 1     # Low risk
}

# --- Compute Material_Risk_Index ---
# Multiply each one-hot column by its risk value, then sum across rows
dfMarket['material_risk_index'] = sum(
    dfMarket[col] * risk for col, risk in risk_map.items()
)

"""
# feature 6:. Stress Index
dfMarket['Stress_Index'] = dfMarket['max_pressure_bar'] / (dfMarket['thickness_mm'] / dfMarket['pipe_size_mm'])

# feature 7. Temp Factor
dfMarket['Temp_Factor'] = np.exp(dfMarket['temperature_c'] / 100)

# feature 8: Temp to Pressure
dfMarket['Temp_to_Pressure'] = dfMarket['temperature_c'] / dfMarket['max_pressure_bar']

# feature 9: Operational Stress Age
dfMarket['Operational_Stress_Age'] = dfMarket['max_pressure_bar'] * dfMarket['time_years']
"""

#FEATURE SCALING
#using Standard Scaler for material loss percent column only
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(dfMarket[["material_loss_percent"]])
material_loss_percent_ss = ss.transform(dfMarket[["material_loss_percent"]])
dfMarket["material_loss_percent_ss"] = material_loss_percent_ss

#FUNCTION TRANSFORMER
from sklearn.preprocessing import FunctionTransformer
ft = FunctionTransformer(func = np.log1p)

ft.fit(dfMarket[["pipe_size_mm"]])
ft.transform(dfMarket[["pipe_size_mm"]])

dfMarket["pipe_size_mm_tf"] = ft.transform(dfMarket[["pipe_size_mm"]])

#TRAIN AND TEST
input_cols = [
    'thickness_mm', 'max_pressure_psi', 'temperature_c',
    'corrosion_impact_percent', 'thickness_loss_mm',
    'time_years',
    'material_carbon steel', 'material_fiberglass', 'material_hdpe',
    'material_pvc', 'material_stainless steel',
    'material_loss_percent_ss', 'pipe_size_mm_tf',
    'corrosion_rate_mm_per_year', 'pressure_to_thickness', 'loss_ratio',
    'material_risk_index'
]

input_data = dfMarket[input_cols]
output_data = dfMarket["condition_encoded"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size = 0.25)

#XGBOOST
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize model
xgb = XGBClassifier(
    n_estimators=200,       # number of trees
    learning_rate=0.05,     # smaller = slower but better generalization
    max_depth=6,            # depth of each tree
    subsample=0.8,          # sample 80% of data for each tree
    colsample_bytree=0.8,   # sample 80% of features for each tree
    random_state=42
)

# Train the model
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb, input_data, output_data, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


