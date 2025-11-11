import pandas as pd
import numpy as np


def prepare_features(df, model=None):
    """
    Prepare features for prediction with exact model alignment
    """

    # --- 1. Handle one-hot encoding for 'material' ---
    materials = ['carbon steel', 'fiberglass', 'hdpe', 'pvc', 'stainless steel']
    for mat in materials:
        col_name = f"material_{mat.replace(' ', '_')}"
        df[col_name] = (df['material'].str.lower() == mat).astype(int)

    # --- 2. Derived / engineered features ---
    df['loss_ratio'] = df['material_loss_percent'] / (df['corrosion_impact_percent'] + 1e-6)
    df['normalized_thickness'] = df['thickness_mm'] / df['pipe_size_mm_tf']

    # Handle division by zero for Stress_Index
    denominator = (df['thickness_mm'] / df['pipe_size_mm_tf'])
    df['Stress_Index'] = df['max_pressure_psi'] / (denominator.replace(0, 0.001))

    # --- 3. Material risk index ---
    risk_map = {
        'material_carbon_steel': 3,  # High risk
        'material_fiberglass': 0,  # Very low risk
        'material_hdpe': 0,  # Very low risk
        'material_pvc': 0,  # Very low risk
        'material_stainless_steel': 1  # Low risk
    }
    df['material_risk_index'] = 0
    for col, risk in risk_map.items():
        if col in df.columns:
            df['material_risk_index'] += df[col] * risk

    # --- 4. Interaction and ratio features ---
    df['pressure_stress_interaction'] = df['max_pressure_psi'] * df['Stress_Index']
    df['size_pressure_index'] = df['pipe_size_mm_tf'] * df['max_pressure_psi']
    df['pressure_to_thickness'] = df['max_pressure_psi'] / df['thickness_mm'].replace(0, 0.001)
    df['pressure_to_size'] = df['max_pressure_psi'] / df['pipe_size_mm_tf'].replace(0, 0.001)
    df['thickness_decay_ratio'] = df['thickness_mm'] / df['time_years'].replace(0, 0.001)
    df['pressure_temp_interaction'] = df['max_pressure_psi'] * df['temperature_c']
    df['Operational_Stress_Age'] = df['max_pressure_psi'] * df['time_years']

    # --- 5. Age group binning ---
    bins = [0, 5, 10, 15, 20, 25, float('inf')]
    labels = ['1–5 years', '6–10 years', '11–15 years', '16–20 years', '21–25 years', '25+ years']
    df['age_group'] = pd.cut(df['time_years'], bins=bins, labels=labels, right=True)
    age_group_encoded = pd.get_dummies(df['age_group'], prefix='age_group').astype(int)
    df = pd.concat([df, age_group_encoded], axis=1)

    # --- 6. Critical: Exact feature alignment with model ---
    if model is not None and hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_

        # Add missing features with default value 0
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
                print(f"⚠️ Added missing feature: {feature}")

        # Remove extra features not used by model
        extra_features = set(df.columns) - set(expected_features)
        if extra_features:
            print(f"⚠️ Removing extra features: {extra_features}")
            df = df[expected_features]

        # Ensure correct order
        df = df[expected_features]

    return df


def validate_input_ranges(user_input):
    """
    Validate input ranges and return warnings for edge cases
    """
    warnings = []

    # Define typical ranges from training data
    typical_ranges = {
        'thickness_mm': (5, 50),
        'corrosion_impact_percent': (0, 30),
        'material_loss_percent': (0, 20),
        'time_years': (0, 30),
        'max_pressure_psi': (50, 500)
    }

    for param, (min_val, max_val) in typical_ranges.items():
        if param in user_input.columns:
            value = user_input[param].iloc[0]
            if value < min_val or value > max_val:
                warnings.append(f"{param}: {value} is outside typical range ({min_val}-{max_val})")

    # Specific business logic checks
    if 'corrosion_impact_percent' in user_input.columns:
        corrosion = user_input['corrosion_impact_percent'].iloc[0]
        if corrosion > 25:
            warnings.append("High corrosion impact - model may be less accurate")

    if 'thickness_mm' in user_input.columns:
        thickness = user_input['thickness_mm'].iloc[0]
        if thickness < 6:
            warnings.append("Very thin pipe - extreme case requiring verification")

    return warnings