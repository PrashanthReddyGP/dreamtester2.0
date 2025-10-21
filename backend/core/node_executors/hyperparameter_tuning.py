# ==============================================================================
# Standard Library Imports
# ==============================================================================
import ast
from typing import Dict, Any
import pandas as pd
import json

# ==============================================================================
# Third-party Library Imports
# ==============================================================================
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
from core.ml_models import get_model

# ==============================================================================
# Main Executor Class
# ==============================================================================

class HyperparameterTuningExecutor(BaseNodeExecutor):
    """
    Executes hyperparameter tuning for a specified ML model.

    This node uses GridSearchCV or RandomizedSearchCV to find the best set of
    hyperparameters from a given search space (param_grid). It then stores
    the best-performing model and detailed tuning results in the context.
    """

    def execute(
        self,
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        """
        Core execution method for the hyperparameter tuning node.
        """
        print(f"Executing Hyperparameter Tuning Node: {node.data.get('label', node.id)}")

        # --- 1. Input Validation ---
        if not inputs:
            raise ValueError("HyperparameterTuning node requires an input connection.")
        
        input_node_id = list(inputs.keys())[0]
        df_input = inputs[input_node_id]['data'].copy()

        if 'label' not in df_input.columns:
            raise ValueError("Input data for HyperparameterTuning must contain a 'label' column.")

        # --- 2. Configuration ---
        config = node.data
        model_name = config.get('modelName')
        strategy = config.get('searchStrategy', 'GridSearchCV')
        cv_folds = int(config.get('cvFolds', 5))
        scoring = config.get('scoringMetric', 'accuracy')
        param_grid_str = config.get('paramGrid', {})
        
        # A value > 0 will print progress updates to the terminal.
        # A higher value (e.g., 2 or 3) provides more detail.
        verbosity = int(config.get('verbosity', 2))
        
        # More robust metric construction logic
        scoring_base = config.get('scoringMetricBase', 'accuracy')
        scoring_avg = config.get('scoringMetricAvg', '') # e.g., 'weighted', 'per-class'
        scoring_class = config.get('scoringMetricClass', '') # e.g., '1', '0'

        # Check if the base metric requires averaging
        if scoring_base in ['f1', 'precision', 'recall']:
            if not scoring_avg:
                raise ValueError(f"Metric '{scoring_base}' requires an averaging method ('weighted', 'macro', etc.).")
            
            if scoring_avg == 'per-class':
                if not scoring_class:
                    raise ValueError("Averaging 'per-class' requires a specific class label to be provided.")
                # Format: 'f1_1', 'precision_0'
                scoring = f"{scoring_base}_{scoring_class}"
            else:
                # Format: 'f1_weighted', 'precision_macro'
                scoring = f"{scoring_base}_{scoring_avg}"
        else:
            # For metrics like 'accuracy', 'r2', 'roc_auc_ovr'
            scoring = scoring_base
        
        print(f"Constructed final scoring metric: '{scoring}'")

        if not model_name:
            raise ValueError("HyperparameterTuning node requires a 'modelName'.")
        if not param_grid_str:
            raise ValueError("HyperparameterTuning node requires a 'paramGrid'.")

        # --- 3. Parse Parameter Grid ---
        # Safely evaluate string representations of lists/ranges into Python objects
        param_grid = {}
        for key, value_str in param_grid_str.items():
            try:
                # ast.literal_eval is a safe way to evaluate a string containing a Python literal
                param_grid[key] = ast.literal_eval(value_str)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid format in param_grid for '{key}': {value_str}. Error: {e}")

        # --- 4. Data Preparation ---
        df_labeled = df_input.dropna(subset=['label']).copy()
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
        feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]

        if not feature_cols:
            raise ValueError("No feature columns found for tuning.")

        X = df_labeled[feature_cols]
        y = df_labeled['label']

        if scoring_avg == 'per-class' and scoring_class:
            unique_labels = y.unique().astype(str)
            if scoring_class not in unique_labels:
                raise ValueError(
                    f"The specified class label '{scoring_class}' for scoring "
                    f"was not found in the 'label' column. Available labels: {list(unique_labels)}"
                )

        # --- 5. Setup and Run Search ---
        base_model = get_model(model_name, {}) # Get a model instance with default params

        if strategy == 'GridSearchCV':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1, # Use all available cores
                verbose=verbosity,
            )
        elif strategy == 'RandomizedSearchCV':
            # n_iter controls how many random combinations are tried
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=10, # A common default, could be a user setting
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=verbosity,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported search strategy: {strategy}")

        print(f"Starting {strategy} for {model_name}...")
        
        search.fit(X, y)
        
        print("Hyperparameter tuning complete.")

        # --- 6. Store Results in Context ---
        best_model = search.best_estimator_
        
        # Make cv_results JSON serializable
        cv_results_df = pd.DataFrame(search.cv_results_)
        # Select most relevant columns for display
        display_cols = ['params'] + [col for col in cv_results_df.columns if 'test_score' in col]
        cv_results_serializable = cv_results_df[display_cols].to_dict(orient='records')


        metadata_payload = {
            "tuning_summary": {
                "Best Score": round(search.best_score_, 4),
                "Best Parameters": search.best_params_,
                "Model": model_name,
                "Strategy": strategy,
                "CV Folds": cv_folds,
                "Scoring Metric": scoring,
            },
            "cv_results": cv_results_serializable,
        }

        context.node_metadata[node.id] = metadata_payload
        context.trained_models[node.id] = best_model # Store the BEST found model
        print(f"Stored best model and tuning results for node {node.id}")

        # --- 7. Return Output DataFrame ---
        return {"default": df_input}
