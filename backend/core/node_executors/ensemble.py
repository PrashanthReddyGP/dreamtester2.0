from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput
from sklearn.ensemble import VotingClassifier
# from your_model_utils import get_model, calculate_model_performance

class EnsembleModelExecutor(BaseNodeExecutor):
    def execute(self, node, inputs, context):
        print(f"Executing Ensemble Model Node: {node.id}")
        df_input = list(inputs.values())[0].copy()

        # Config for the ensemble, defined in the node's data property
        # e.g., node.data['models'] = [
        #   {'name': 'LogisticRegression', 'params': {'C': 1.0}},
        #   {'name': 'RandomForestClassifier', 'params': {'n_estimators': 50}}
        # ]
        model_configs = node.data.get('models', [])
        ensemble_type = node.data.get('ensembleType', 'voting') # 'voting' or 'stacking'

        if not model_configs:
            raise ValueError("Ensemble node requires at least one model configuration.")

        # --- Prepare Data ---
        # (Same as ModelTrainerExecutor)
        df_labeled = df_input.dropna(subset=['label']).copy()
        # ... X, y, X_train, ... setup
        
        # --- Create Ensemble ---
        estimators = []
        for i, config in enumerate(model_configs):
            # Ensure unique names for estimators
            estimator_name = f"{config['name'].lower()}_{i}"
            estimators.append((estimator_name, get_model(config['name'], config.get('params', {}))))
        
        if ensemble_type == 'voting':
            ensemble_model = VotingClassifier(estimators=estimators, voting='soft') # soft voting is often better
        # elif ensemble_type == 'stacking':
        #   ensemble_model = StackingClassifier(estimators=estimators, final_estimator=...)
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

        # --- Train & Evaluate ---
        ensemble_model.fit(X_train, y_train)
        metrics, analysis = calculate_model_performance(ensemble_model, X_test, y_test, X.columns)
        
        # --- Store results ---
        context.node_metadata[node.id] = { "model_metrics": metrics, "model_analysis": analysis }
        context.trained_models[node.id] = ensemble_model
        
        return df_input