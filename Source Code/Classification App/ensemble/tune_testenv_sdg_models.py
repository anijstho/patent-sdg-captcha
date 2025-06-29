#!/usr/bin/env python3
"""
SDG Model Hyperparameter Tuning Script (Testenv Version)

This script performs hyperparameter tuning for the XGBoost ensemble model 
using data from testenv_captcha_labeled2.pkl.

It maintains the binary label conversion (threshold 0.5) and sample weighting
(abs(sdgXX.gt-0.5)*2*sdgXX.n) approach from the testenv analysis.
"""

import pickle
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_testenv(file_path='testenv_captcha_labeled2.pkl'):
    """Load the testenv DataFrame from the specified file."""
    print(f"Loading testenv data from {file_path}...")
    with open(file_path, 'rb') as f:
        testenv = pickle.load(f)
    
    print(f"Loaded testenv with {len(testenv.index)} rows")
    return testenv


def extract_data_for_sdg_tuning(testenv, sdg_num):
    """
    Extract model predictions, ground truth labels, and weights for a specific SDG.
    Convert ground truth values to binary using a 0.5 threshold.
    Calculate sample weights using abs(sdgXX.gt-0.5)*2*sdgXX.n
    
    Args:
        testenv: The testenv DataFrame
        sdg_num: SDG number (1-16)
        
    Returns:
        Dictionary of model predictions, ground truth labels, and sample weights
    """
    # Format SDG number with leading zero
    sdg_key = f"sdg{sdg_num:02d}"
    
    # Get ground truth column
    gt_col = f"{sdg_key}.gt"
    n_col = f"{sdg_key}.n"
    
    # Filter rows where gt and n are not missing
    valid_rows = ~testenv[gt_col].isna() & ~testenv[n_col].isna()
    filtered_testenv = testenv.loc[valid_rows].copy()
    
    # Skip if no valid rows
    if len(filtered_testenv) == 0:
        print(f"No valid ground truth data found for {sdg_key}")
        return None
    
    # Convert ground truth to binary (exclude exactly 0.5)
    gt_values = filtered_testenv[gt_col].values
    binary_gt = np.zeros(len(gt_values), dtype=int)
    
    # Set to 1 if gt > 0.5, keep as 0 if gt < 0.5, exclude if gt == 0.5
    positive_mask = gt_values > 0.5
    negative_mask = gt_values < 0.5
    valid_mask = positive_mask | negative_mask
    
    # If no valid samples after excluding 0.5 values, skip
    if not np.any(valid_mask):
        print(f"No valid ground truth data after excluding 0.5 values for {sdg_key}")
        return None
    
    # Apply binary conversion only to valid samples
    filtered_testenv = filtered_testenv[valid_mask].copy()
    gt_values = gt_values[valid_mask]
    binary_gt = np.zeros(len(gt_values), dtype=int)
    binary_gt[gt_values > 0.5] = 1
    
    # Calculate weights: abs(gt-0.5)*2*n
    n_values = filtered_testenv[n_col].values
    weights = np.abs(gt_values - 0.5) * 2 * n_values
    
    # Get model predictions for this SDG
    model_predictions = {}
    model_names = ['auckland', 'aurora', 'elsevier', 'sdgo', 'sdsn', 'siris', 'sdgbert', 'mistral']
    
    for model in model_names:
        model_col = f"{sdg_key}.{model}"
        if model_col in filtered_testenv.columns:
            # Note: sdgbert values are already binary
            model_predictions[model] = filtered_testenv[model_col].values.astype(int)
        else:
            print(f"Warning: {model_col} not found in testenv")
    
    # Create features matrix
    model_names = list(model_predictions.keys())
    X = np.column_stack([model_predictions[model] for model in model_names])
    
    return {
        'X': X,
        'y': binary_gt,
        'weights': weights,
        'model_predictions': model_predictions,
        'model_names': model_names
    }


def evaluate_default_model(X, y, weights=None):
    """Evaluate the default model using cross-validation."""
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Initialize array to store predictions
    default_predictions = np.zeros(len(y))
    
    # Perform cross-validation
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get weights for training samples if provided
        train_weights = None
        if weights is not None:
            train_weights = weights[train_idx]
        
        # Initialize model with default parameters
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_SEED
        )
        
        # Fit model with or without weights
        if train_weights is not None:
            model.fit(X_train, y_train, sample_weight=train_weights)
        else:
            model.fit(X_train, y_train)
        
        # Make predictions for this fold
        default_predictions[test_idx] = model.predict(X_test)
    
    # Calculate F1 score
    f1 = f1_score(y, default_predictions, average='binary')
    return f1, default_predictions


def tune_ensemble_model(X, y, weights, sdg_name):
    """
    Tune hyperparameters for XGBoost ensemble model using grid search.
    
    Args:
        X: Features matrix (predictions from the individual models)
        y: Target vector (ground truth binary labels)
        weights: Sample weights
        sdg_name: Name of the SDG being processed (for logging)
        
    Returns:
        Dictionary with best parameters and F1 score
    """
    print(f"\nTuning hyperparameters for {sdg_name}...")
    start_time = time()
    
    # Calculate class distribution for scale_pos_weight
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Class distribution - Positive: {pos_count}, Negative: {neg_count}")
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1.0]
    }
    
    # Base XGBoost classifier with fixed parameters
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight
    )
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Create F1 scorer
    f1_scorer = make_scorer(f1_score)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=skf,
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    
    # Run grid search
    try:
        # Use weights if provided
        if weights is not None:
            grid_search.fit(X, y, sample_weight=weights)
        else:
            grid_search.fit(X, y)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters for {sdg_name}: {best_params}")
        print(f"Best F1 score for {sdg_name}: {best_score:.4f}")
        
        # Create and train a model with the best parameters
        best_model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            scale_pos_weight=scale_pos_weight,
            **best_params
        )
        
        # Train on full dataset
        if weights is not None:
            best_model.fit(X, y, sample_weight=weights)
        else:
            best_model.fit(X, y)
        
        # Save the trained model
        model_filename = f"models/{sdg_name}_testenv_model.pkl"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"Model saved to {model_filename}")
        
        elapsed_time = time() - start_time
        print(f"Tuning completed in {elapsed_time:.2f} seconds")
        
        # Return results
        return {
            'best_params': best_params,
            'best_score': best_score,
            'model_filename': model_filename
        }
        
    except Exception as e:
        print(f"Error during grid search for {sdg_name}: {e}")
        return {
            'best_params': None,
            'best_score': None,
            'error': str(e)
        }


def evaluate_tuned_model(X, y, weights, tuned_params, sdg_name):
    """
    Evaluate the tuned model using cross-validation.
    
    Args:
        X: Features matrix
        y: Target vector
        weights: Sample weights
        tuned_params: Dictionary with best parameters from grid search
        sdg_name: Name of the SDG being processed
        
    Returns:
        Predictions from the tuned ensemble model
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Initialize array to store predictions
    tuned_predictions = np.zeros(len(y))
    
    # Calculate class distribution for scale_pos_weight
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Perform cross-validation
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get weights for training samples if provided
        train_weights = None
        if weights is not None:
            train_weights = weights[train_idx]
        
        # Initialize model with tuned parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            scale_pos_weight=scale_pos_weight,
            **tuned_params['best_params']
        )
        
        # Fit model with or without weights
        if train_weights is not None:
            model.fit(X_train, y_train, sample_weight=train_weights)
        else:
            model.fit(X_train, y_train)
        
        # Make predictions for this fold
        tuned_predictions[test_idx] = model.predict(X_test)
    
    # Calculate F1 score
    f1 = f1_score(y, tuned_predictions, average='binary')
    return f1, tuned_predictions


def process_all_sdgs(testenv):
    """
    Process all SDGs in the testenv and tune hyperparameters.
    
    Args:
        testenv: The testenv DataFrame
        
    Returns:
        Dictionary with tuning results for each SDG
    """
    all_results = {}
    
    # Process each SDG
    for i in range(1, 17):
        sdg_num = i
        sdg_name = f"sdg{sdg_num:02d}"
        
        print(f"\nProcessing {sdg_name}...")
        
        # Extract data for this SDG
        extracted_data = extract_data_for_sdg_tuning(testenv, sdg_num)
        
        # Skip if no valid data
        if extracted_data is None:
            print(f"Skipping {sdg_name} due to insufficient data")
            all_results[sdg_name] = {'error': 'Insufficient data'}
            continue
        
        try:
            # Evaluate default model
            default_f1, default_preds = evaluate_default_model(
                extracted_data['X'], 
                extracted_data['y'],
                extracted_data['weights']
            )
            print(f"Default model F1 score for {sdg_name}: {default_f1:.4f}")
            
            # Tune ensemble model
            tuning_results = tune_ensemble_model(
                extracted_data['X'], 
                extracted_data['y'],
                extracted_data['weights'],
                sdg_name
            )
            
            if tuning_results['best_params'] is not None:
                # Evaluate the tuned model
                tuned_f1, tuned_preds = evaluate_tuned_model(
                    extracted_data['X'], 
                    extracted_data['y'],
                    extracted_data['weights'],
                    tuning_results,
                    sdg_name
                )
                print(f"Tuned model F1 score for {sdg_name}: {tuned_f1:.4f}")
                
                # Store results
                tuning_results['default_f1'] = default_f1
                tuning_results['tuned_f1'] = tuned_f1
                tuning_results['improvement'] = tuned_f1 - default_f1
                
                # Add model performance to dictionary
                model_predictions = extracted_data['model_predictions']
                ground_truth = extracted_data['y']
                
                model_scores = {}
                for model, preds in model_predictions.items():
                    model_scores[model] = f1_score(ground_truth, preds, average='binary')
                
                # Add ensemble results
                model_scores['default_ensemble'] = default_f1
                model_scores['tuned_ensemble'] = tuned_f1
                
                tuning_results['model_scores'] = model_scores
                
                all_results[sdg_name] = tuning_results
            
        except Exception as e:
            print(f"Error processing {sdg_name}: {e}")
            all_results[sdg_name] = {'error': str(e)}
    
    return all_results


def visualize_improvements(all_results):
    """
    Visualize the improvements from hyperparameter tuning.
    
    Args:
        all_results: Dictionary with tuning results for each SDG
    """
    # Extract data for visualization
    sdgs = []
    default_scores = []
    tuned_scores = []
    
    for sdg, results in all_results.items():
        if 'default_f1' in results and 'tuned_f1' in results:
            sdgs.append(sdg)
            default_scores.append(results['default_f1'])
            tuned_scores.append(results['tuned_f1'])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'SDG': sdgs * 2,
        'F1 Score': default_scores + tuned_scores,
        'Model Type': ['Default'] * len(sdgs) + ['Tuned'] * len(sdgs)
    })
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='SDG', y='F1 Score', hue='Model Type')
    plt.title('F1 Score Comparison: Default vs. Tuned Hyperparameters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('testenv_hyperparameter_tuning_comparison.png')
    plt.close()
    
    # Create heatmap of improvements
    improvements = {}
    for sdg, results in all_results.items():
        if 'improvement' in results:
            improvements[sdg] = results['improvement']
    
    # Sort SDGs by improvement
    sorted_sdgs = sorted(improvements.keys(), key=lambda x: improvements[x], reverse=True)
    improvement_values = [improvements[sdg] for sdg in sorted_sdgs]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(y=sorted_sdgs, x=improvement_values, orient='h')
    plt.title('Improvement in F1 Score from Hyperparameter Tuning')
    plt.xlabel('F1 Score Improvement')
    plt.ylabel('SDG')
    plt.tight_layout()
    plt.savefig('testenv_hyperparameter_improvement.png')
    plt.close()


def create_model_comparison_heatmap(all_results):
    """
    Create a heatmap comparing all models including default and tuned ensemble.
    
    Args:
        all_results: Dictionary with tuning results for each SDG
    """
    # Prepare data for heatmap
    model_data = {}
    sdgs = []
    
    for sdg, results in all_results.items():
        if 'model_scores' in results:
            sdgs.append(sdg)
            for model, score in results['model_scores'].items():
                if model not in model_data:
                    model_data[model] = []
                model_data[model].append(score)
    
    # Create DataFrame for heatmap
    comparison_df = pd.DataFrame(model_data, index=sdgs)
    
    # Add average row
    avg_row = pd.DataFrame(comparison_df.mean().values.reshape(1, -1), 
                          columns=comparison_df.columns, 
                          index=['Average'])
    comparison_df = pd.concat([comparison_df, avg_row])
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(comparison_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
    plt.title('F1 Scores by Model and SDG')
    plt.ylabel('SDG')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig('testenv_model_comparison_heatmap.png')
    plt.close()
    
    return comparison_df


def save_results(all_results, comparison_df):
    """
    Save the tuning results to files.
    
    Args:
        all_results: Dictionary with tuning results for each SDG
        comparison_df: DataFrame with model comparison results
    """
    # Save comparison DataFrame to CSV
    comparison_df.to_csv('testenv_model_comparison.csv')
    print("Model comparison saved to testenv_model_comparison.csv")
    
    # Convert tuning results to DataFrame for easier analysis
    rows = []
    
    for sdg, results in all_results.items():
        if 'best_params' in results and results['best_params'] is not None:
            row = {
                'sdg': sdg,
                'default_f1': results.get('default_f1', None),
                'tuned_f1': results.get('tuned_f1', None),
                'improvement': results.get('improvement', None),
                'best_score': results.get('best_score', None)
            }
            
            # Add best parameters
            for param, value in results['best_params'].items():
                row[f'param_{param}'] = value
                
            rows.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(rows)
    
    # Calculate average improvement
    avg_improvement = results_df['improvement'].mean()
    print(f"\nAverage F1 score improvement: {avg_improvement:.4f}")
    
    # Save to CSV
    results_df.to_csv('testenv_ensemble_tuning_results.csv', index=False)
    print("Tuning results saved to testenv_ensemble_tuning_results.csv")
    
    # Save full results as JSON
    with open('testenv_ensemble_tuning_results_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Full results saved to testenv_ensemble_tuning_results_full.json")


if __name__ == "__main__":
    # Load testenv data
    testenv = load_testenv()
    
    # Process all SDGs
    all_results = process_all_sdgs(testenv)
    
    # Visualize improvements
    visualize_improvements(all_results)
    
    # Create model comparison heatmap
    comparison_df = create_model_comparison_heatmap(all_results)
    
    # Save results
    save_results(all_results, comparison_df)
    
    print("\nProcess completed successfully.")
