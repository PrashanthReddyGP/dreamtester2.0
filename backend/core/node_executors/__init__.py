from typing import Any
from .base import BaseNodeExecutor
from .data_source import DataSourceExecutor
from .feature import FeatureExecutor
from .process_indicators import ProcessIndicatorsExecutor
from .custom_code import CustomCodeExecutor
from .custom_labeling import CustomLabelingExecutor
from .data_scaling import DataScalingExecutor
from .data_validation import DataValidationExecutor
from .model_trainer import ModelTrainerExecutor
from .charting import ChartingExecutor
from .model_predictor import ModelPredictorExecutor
from .merge import MergeExecutor
from .features_correlation import FeaturesCorrelationExecutor
from .hyperparameter_tuning import HyperparameterTuningExecutor
from .class_imbalance import ClassImbalanceExecutor
from .backtester import BacktesterNodeExecutor

# The Registry maps node type strings to executor instances
NODE_EXECUTORS = {
    "dataSource": DataSourceExecutor(),
    "feature": FeatureExecutor(),
    "processIndicators": ProcessIndicatorsExecutor(),
    "customCode": CustomCodeExecutor(),
    "customLabeling": CustomLabelingExecutor(),
    "dataScaling": DataScalingExecutor(),
    "dataValidation": DataValidationExecutor(),
    "charting": ChartingExecutor(),
    "modelTrainer": ModelTrainerExecutor(),
    "hyperparameterTuning": HyperparameterTuningExecutor(),
    "modelPredictor": ModelPredictorExecutor(),
    "merge": MergeExecutor(),
    "featuresCorrelation": FeaturesCorrelationExecutor(),
    "classImbalance": ClassImbalanceExecutor(),
    "backtester": BacktesterNodeExecutor(),
}

def get_executor(node_type: str) -> BaseNodeExecutor:
    """
    Retrieves the executor instance for a given node type.
    """
    executor = NODE_EXECUTORS.get(node_type)
    if not executor:
        raise NotImplementedError(f"No executor found for node type: '{node_type}'")
    return executor