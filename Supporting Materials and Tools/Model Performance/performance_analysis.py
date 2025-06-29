#!/usr/bin/env python3
"""
This script creates and analyzes ensemble models for SDG predictions, as it:

1. Checks for existing tuned ensemble models in the models/ subfolder
2. Creates tuned models if they don't exist by tuning xgboost hyperparameters
3. Analyzes SDG model performance with ensemble variants with and without Mistral
4. Creates the F1 heatmap visualization under f1_heatmap.png
"""

import pickle
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_testenv(file_path='training_data.pkl'):
    """Load the testenv DataFrame from the specified file."""
    print(f"Loading testenv data from {file_path}...")
    try:
        testenv = pd.read_pickle(file_path)
        print(f"Loaded testenv with {len(testenv.index)} rows")
        return testenv
    except Exception as e:
        print(f"Error loading testenv: {e}")
        return None


def check_existing_models():
    """Check which ensemble models already exist in the models/ folder."""
    models_dir = "models"
    existing_models = {}
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return existing_models
    
    for i in range(1, 17):
        sdg_key = f"sdg{i:02d}"
        full_model_path = os.path.join(models_dir, f"{sdg_key}_full_ensemble_model.pkl")
        no_mistral_model_path = os.path.join(models_dir, f"{sdg_key}_no_mistral_model.pkl")
        
        existing_models[sdg_key] = {
            'full_ensemble': os.path.exists(full_model_path),
            'no_mistral': os.path.exists(no_mistral_model_path)
        }
    
    return existing_models


def extract_data_for_sdg(testenv, sdg_num, include_mistral=True):
    """
    Extract model predictions and ground truth for a specific SDG.
    
    Args:
        testenv: The testenv DataFrame
        sdg_num: SDG number (1-16)
        include_mistral: Whether to include Mistral in the feature set
        
    Returns:
        Dictionary with features, labels, and weights
    """
    # Format SDG number with leading zero
    sdg_key = f"sdg{sdg_num:02d}"
    
    # Get ground truth and n columns
    gt_col = f"{sdg_key}.gt"
    n_col = f"{sdg_key}.n"
    
    # Filter rows where gt and n are not missing
    valid_rows = ~testenv[gt_col].isna() & ~testenv[n_col].isna()
    filtered_testenv = testenv.loc[valid_rows].copy()
    
    if len(filtered_testenv) == 0:
        print(f"No valid ground truth data found for {sdg_key}")
        return None
    
    # Convert ground truth to binary (exclude exactly 0.5)
    gt_values = filtered_testenv[gt_col].values
    valid_mask = (gt_values > 0.5) | (gt_values < 0.5)
    
    if not np.any(valid_mask):
        print(f"No valid ground truth data after excluding 0.5 values for {sdg_key}")
        return None
    
    # Apply filter
    filtered_testenv = filtered_testenv[valid_mask].copy()
    gt_values = gt_values[valid_mask]
    binary_gt = (gt_values > 0.5).astype(int)
    
    # Calculate weights
    n_values = filtered_testenv[n_col].values
    weights = np.abs(gt_values - 0.5) * 2 * n_values
    
    # Get model predictions
    model_names = ['auckland', 'aurora', 'elsevier', 'sdgo', 'sdsn', 'siris', 'sdgbert']
    if include_mistral:
        model_names.append('mistral')
    
    model_predictions = {}
    for model in model_names:
        model_col = f"{sdg_key}.{model}"
        if model_col in filtered_testenv.columns:
            model_predictions[model] = filtered_testenv[model_col].values.astype(int)
    
    # Create features matrix
    available_models = list(model_predictions.keys())
    X = np.column_stack([model_predictions[model] for model in available_models])
    
    return {
        'X': X,
        'y': binary_gt,
        'weights': weights,
        'model_predictions': model_predictions,
        'model_names': available_models
    }


def load_or_create_tuned_params(sdg_key, include_mistral=True):
    """Load tuned parameters from file or return None if not available."""
    if include_mistral:
        param_file = f"tuned_params_{sdg_key}.json"
    else:
        param_file = f"tuned_params_{sdg_key}_no_mistral.json"
    
    if os.path.exists(param_file):
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
            return params
        except:
            return None
    return None


def tune_ensemble_hyperparameters(X, y, weights, sdg_key, include_mistral=True):
    """Tune hyperparameters for the ensemble model."""
    print(f"Tuning hyperparameters for {sdg_key} ({'full' if include_mistral else 'no-mistral'} ensemble)...")
    
    # Calculate class distribution
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0]
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight
    )
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    f1_scorer = make_scorer(f1_score)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=skf,
        n_jobs=-1
    )
    
    grid_search.fit(X, y, sample_weight=weights)
    
    best_params = grid_search.best_params_
    print(f"Best parameters for {sdg_key}: {best_params}")
    
    # Save parameters
    suffix = "" if include_mistral else "_no_mistral"
    param_file = f"tuned_params_{sdg_key}{suffix}.json"
    with open(param_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    return best_params


def train_ensemble_model(testenv, sdg_num, model_type='full'):
    """Train an ensemble model for a specific SDG."""
    sdg_key = f"sdg{sdg_num:02d}"
    include_mistral = (model_type == 'full')
    
    # Extract data
    data = extract_data_for_sdg(testenv, sdg_num, include_mistral=include_mistral)
    if data is None:
        return None
    
    # Load or tune parameters
    tuned_params = load_or_create_tuned_params(sdg_key, include_mistral)
    if tuned_params is None:
        tuned_params = tune_ensemble_hyperparameters(
            data['X'], data['y'], data['weights'], sdg_key, include_mistral
        )
    else:
        params_str = ', '.join([f"{k}={v}" for k, v in tuned_params.items()])
        suffix = "" if include_mistral else " (no Mistral)"
        print(f"Loaded tuned parameters for {sdg_key}{suffix}: {params_str}")
    
    # Calculate scale_pos_weight
    pos_count = np.sum(data['y'] == 1)
    neg_count = np.sum(data['y'] == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Train final model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        **tuned_params
    )
    
    model.fit(data['X'], data['y'], sample_weight=data['weights'])
    
    # Save model
    model_filename = f"models/{sdg_key}_{'full_ensemble' if include_mistral else 'no_mistral'}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    return model


def calculate_model_metrics(model_predictions, ground_truth):
    """Calculate metrics for all models."""
    metrics = {}
    
    for model, predictions in model_predictions.items():
        f1 = f1_score(ground_truth, predictions, average='binary')
        metrics[model] = {'f1_score': f1}
    
    return metrics


def analyze_testenv_sdgs(testenv, output_format='dataframe'):
    """
    Analyze testenv data and return F1 scores for all models including ensembles.
    
    Args:
        testenv: The testenv DataFrame
        output_format: 'dataframe' or 'dict'
        
    Returns:
        DataFrame with F1 scores or dictionary with detailed results
    """
    all_results = {}
    existing_models = check_existing_models()
    
    # Process each SDG
    for i in range(1, 17):
        sdg_num = i
        sdg_key = f"sdg{sdg_num:02d}"
        
        print(f"\nProcessing {sdg_key}...")
        
        # Check if we need to create ensemble models
        need_full_ensemble = not existing_models[sdg_key]['full_ensemble']
        need_no_mistral = not existing_models[sdg_key]['no_mistral']
        
        # Create models if needed
        if need_full_ensemble:
            print(f"Creating full ensemble model for {sdg_key}...")
            train_ensemble_model(testenv, sdg_num, 'full')
        
        if need_no_mistral:
            print(f"Creating no-mistral ensemble model for {sdg_key}...")
            train_ensemble_model(testenv, sdg_num, 'no_mistral')
        
        # Extract data for analysis
        data_full = extract_data_for_sdg(testenv, sdg_num, include_mistral=True)
        data_no_mistral = extract_data_for_sdg(testenv, sdg_num, include_mistral=False)
        
        if data_full is None:
            print(f"Skipping {sdg_key} due to insufficient data")
            continue
        
        # Load ensemble models
        full_model_path = f"models/{sdg_key}_full_ensemble_model.pkl"
        no_mistral_model_path = f"models/{sdg_key}_no_mistral_model.pkl"
        
        model_predictions = data_full['model_predictions'].copy()
        ground_truth = data_full['y']
        
        # Add ensemble predictions
        if os.path.exists(full_model_path):
            with open(full_model_path, 'rb') as f:
                full_model = pickle.load(f)
            ensemble_preds = full_model.predict(data_full['X'])
            model_predictions['ensemble'] = ensemble_preds
        
        if os.path.exists(no_mistral_model_path) and data_no_mistral is not None:
            with open(no_mistral_model_path, 'rb') as f:
                no_mistral_model = pickle.load(f)
            no_mistral_preds = no_mistral_model.predict(data_no_mistral['X'])
            model_predictions['ensemble_w/o_mistral'] = no_mistral_preds
        
        # Calculate metrics
        metrics = calculate_model_metrics(model_predictions, ground_truth)
        all_results[sdg_key] = metrics
        
        print(f"Processed {sdg_key}")
    
    # Convert results to the desired format
    if output_format == 'dataframe':
        # Create a multi-index dataframe
        rows = []
        for sdg, models in all_results.items():
            for model, metrics in models.items():
                rows.append({
                    'sdg': sdg,
                    'model': model,
                    'f1_score': metrics['f1_score']
                })
        
        df = pd.DataFrame(rows)
        
        # Pivot the dataframe to have models as columns and SDGs as rows
        pivot_df = df.pivot(index='sdg', columns='model', values='f1_score')
        
        # Reorder columns to match the order in model_predictions dictionary
        # Note: Switched order of ensemble columns as requested
        model_order = ['auckland', 'aurora', 'elsevier', 'sdgo', 'sdsn', 'siris', 'sdgbert', 
                      'mistral', 'ensemble', 'ensemble_w/o_mistral']
        # Only include columns that exist
        available_models = [m for m in model_order if m in pivot_df.columns]
        pivot_df = pivot_df[available_models]
        
        # Add 'best' column showing highest F1 score per row (excluding this calculation from the Average row)
        pivot_df['best'] = pivot_df.max(axis=1)
        
        # Add a row with average F1 scores at the bottom
        avg_row = pd.DataFrame(pivot_df.mean().values.reshape(1, -1), 
                              columns=pivot_df.columns, 
                              index=['avg.'])
        
        # For the 'best' column in the Average row, calculate the average of the best scores
        # (excluding the Average row itself since it doesn't exist yet)
        best_scores = pivot_df['best']  # This includes all SDG rows
        avg_row.loc['avg.', 'best'] = best_scores.mean()
        
        pivot_df = pd.concat([pivot_df, avg_row])
        
        return pivot_df
    else:
        return all_results


def create_heatmap_visualization(testenv):
    """
    Create the main heatmap visualization showing all model performances.
    
    Args:
        testenv: DataFrame containing testenv data
    
    Returns:
        DataFrame with F1 scores for all models and ensembles
    """
    print("Analyzing SDG model performance with ensemble models...")
    
    # Get the results directly from the analyze_testenv_sdgs function
    results_df = analyze_testenv_sdgs(testenv)
    
    # Create a copy for display with renamed column label
    display_df = results_df.copy()
    if 'ensemble_w/o_mistral' in display_df.columns:
        display_df = display_df.rename(columns={'ensemble_w/o_mistral': 'w/o_mistral'})
    
    # Create heatmap visualization
    plt.figure(figsize=(10, 8))  # Made wider to accommodate the new 'best' column
    sns.heatmap(display_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
    plt.title('F1 Scores by Model and SDG')
    #plt.ylabel('SDG')
    plt.xlabel('')
    plt.tight_layout()
    
    # Save the heatmap
    output_path = "f1_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created heatmap visualization: {output_path}")
    
    return results_df


if __name__ == "__main__":
    # Load testenv data
    testenv = load_testenv()
    
    if testenv is not None:
        # Create heatmap visualization
        results = create_heatmap_visualization(testenv)
        
        # Print summary of results
        print("\nAverage F1 scores across all SDGs:")
        print(results.loc['avg.'].sort_values(ascending=False))
        
        print("\nAnalysis complete. Heatmap saved as 'f1_heatmap.png'")
    else:
        print("Failed to load testenv data.")
