// src/components/machinelearning/tabs/TrainingBacktestingTab.tsx
import React, { useEffect, useMemo } from 'react';
import { Box, Typography, Stack, Paper, FormControl, InputLabel, Select, MenuItem, TextField, Button, CircularProgress, Switch, Divider, FormControlLabel } from '@mui/material';
import type { MLConfig } from '../types';
import type { StrategyResult, MLResult } from '../../../services/api';
import { FeatureImportanceChart } from '../../analysishub/FeatureImportanceChart';
import { ClassificationReportTable } from '../../analysishub/ClassificationReportTable';
import { ConfusionMatrix } from '../../analysishub/ConfusionMatrix';

// A type alias for clarity
type AnalysisResult = StrategyResult | MLResult;

interface TrainingTabProps {
    config: MLConfig;
    onChange: (path: string, value: any) => void;
    onRun: () => void;
    isRunning: boolean;
    models: string[];
    result: AnalysisResult,
}

// Configuration for rendering hyperparameter inputs dynamically
const HYPERPARAMETER_CONFIG: Record<string, { name: string; label: string; type: 'number' | 'text' | 'select'; options?: string[]; defaultValue: any; }[]> = {
    'LogisticRegression': [
        { name: 'C', label: 'Inverse Regularization (C)', type: 'number', defaultValue: 1.0 },
        { name: 'penalty', label: 'Penalty', type: 'select', options: ['l1', 'l2', 'elasticnet', 'none'], defaultValue: 'l2' },
        { name: 'solver', label: 'Solver', type: 'select', options: ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], defaultValue: 'lbfgs' },
    ],
    'RandomForestClassifier': [
        { name: 'n_estimators', label: 'Number of Trees', type: 'number', defaultValue: 100 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 10 },
        { name: 'min_samples_split', label: 'Min Samples to Split', type: 'number', defaultValue: 2 },
        { name: 'min_samples_leaf', label: 'Min Samples per Leaf', type: 'number', defaultValue: 1 },
        { name: 'class_weight', label: 'Class Weight', type: 'select', options: ['balanced', 'balanced_subsample', 'none'], defaultValue: 'balanced' },
    ],
    'LightGBMClassifier': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', defaultValue: 0.1 },
        { name: 'num_leaves', label: 'Number of Leaves', type: 'number', defaultValue: 31 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: -1 },
    ],
    'XGBoostClassifier': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', defaultValue: 0.1 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 3 },
        { name: 'subsample', label: 'Subsample Ratio', type: 'number', defaultValue: 1.0 },
        { name: 'colsample_bytree', label: 'Colsample by Tree', type: 'number', defaultValue: 1.0 },
    ],

    'SVC': [ // Support Vector Classifier
        { name: 'C', label: 'Regularization (C)', type: 'number', defaultValue: 1.0 },
        { name: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], defaultValue: 'rbf' },
        { name: 'gamma', label: 'Kernel Coefficient (gamma)', type: 'select', options: ['scale', 'auto'], defaultValue: 'scale' },
    ],
    'KNeighborsClassifier': [
        { name: 'n_neighbors', label: 'Number of Neighbors (K)', type: 'number', defaultValue: 5 },
        { name: 'weights', label: 'Weighting', type: 'select', options: ['uniform', 'distance'], defaultValue: 'uniform' },
        { name: 'algorithm', label: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], defaultValue: 'auto' },
    ],
    'DecisionTreeClassifier': [
        { name: 'criterion', label: 'Criterion', type: 'select', options: ['gini', 'entropy'], defaultValue: 'gini' },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 10 },
        { name: 'min_samples_split', label: 'Min Samples to Split', type: 'number', defaultValue: 2 },
        { name: 'min_samples_leaf', label: 'Min Samples per Leaf', type: 'number', defaultValue: 1 },
    ],

    // --- Regression Models ---
    'LinearRegression': [
        { name: 'fit_intercept', label: 'Fit Intercept', type: 'select', options: ['true', 'false'], defaultValue: 'true' },
    ],
    'RandomForestRegressor': [
        { name: 'n_estimators', label: 'Number of Trees', type: 'number', defaultValue: 100 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 10 },
        { name: 'min_samples_split', label: 'Min Samples to Split', type: 'number', defaultValue: 2 },
        { name: 'criterion', label: 'Criterion', type: 'select', options: ['squared_error', 'absolute_error', 'poisson'], defaultValue: 'squared_error' },
    ],
    'XGBoostRegressor': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', defaultValue: 0.1 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', defaultValue: 3 },
        { name: 'subsample', label: 'Subsample Ratio', type: 'number', defaultValue: 1.0 },
    ],
    'SVR': [ // Support Vector Regressor
        { name: 'C', label: 'Regularization (C)', type: 'number', defaultValue: 1.0 },
        { name: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly'], defaultValue: 'rbf' },
        { name: 'epsilon', label: 'Epsilon (margin of tolerance)', type: 'number', defaultValue: 0.1 },
    ],

    // --- Unsupervised Learning Models ---
    'KMeans': [
        { name: 'n_clusters', label: 'Number of Clusters (K)', type: 'number', defaultValue: 3 },
        { name: 'init', label: 'Initialization Method', type: 'select', options: ['k-means++', 'random'], defaultValue: 'k-means++' },
        { name: 'n_init', label: 'Number of Initializations', type: 'number', defaultValue: 10 },
        { name: 'max_iter', label: 'Max Iterations', type: 'number', defaultValue: 300 },
    ],
    'PCA': [ // Principal Component Analysis
        { name: 'n_components', label: 'Number of Components', type: 'number', defaultValue: 2 },
        { name: 'svd_solver', label: 'SVD Solver', type: 'select', options: ['auto', 'full', 'arpack', 'randomized'], defaultValue: 'auto' },
    ],
    'IsolationForest': [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', defaultValue: 100 },
        { name: 'contamination', label: 'Contamination (Outlier %)', type: 'number', defaultValue: 0.1 },
        { name: 'max_features', label: 'Max Features per Tree', type: 'number', defaultValue: 1.0 },
    ],
    'DBSCAN': [
        { name: 'eps', label: 'Max Distance (eps)', type: 'number', defaultValue: 0.5 },
        { name: 'min_samples', label: 'Min Samples in Neighborhood', type: 'number', defaultValue: 5 },
        { name: 'algorithm', label: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], defaultValue: 'auto' },
    ]
};

export const TrainingBacktestingTab: React.FC<TrainingTabProps> = ({ config, onChange, onRun, isRunning, models, result }) => {
    
    const modelParams = useMemo(() => HYPERPARAMETER_CONFIG[config.model.name] || [], [config.model.name]);
    const isMLResult = result && 'model_analysis' in result;

    // Effect to set default hyperparameters for the selected model if they aren't already set
    useEffect(() => {
        const currentParams = config.model.hyperparameters || {};
        let needsUpdate = false;
        const newParams = { ...currentParams };

        modelParams.forEach(param => {
            if (currentParams[param.name] === undefined) {
                newParams[param.name] = param.defaultValue;
                needsUpdate = true;
            }
        });

        if (needsUpdate) {
            onChange('model.hyperparameters', newParams);
        }
    }, [config.model.name, config.model.hyperparameters, modelParams, onChange]);

    const renderHyperparameterInput = (param: typeof modelParams[0]) => {
        const value = config.model.hyperparameters?.[param.name] ?? param.defaultValue;

        if (param.type === 'select') {
            return (
                <FormControl key={param.name} fullWidth size="small">
                    <InputLabel>{param.label}</InputLabel>
                    <Select
                        value={value}
                        label={param.label}
                        onChange={(e) => onChange(`model.hyperparameters.${param.name}`, e.target.value)}
                    >
                        {param.options?.map(opt => <MenuItem key={opt} value={opt === 'none' ? '' : opt}>{opt}</MenuItem>)}
                    </Select>
                </FormControl>
            );
        }
        return (
            <TextField
                key={param.name}
                label={param.label}
                type={param.type}
                size="small"
                fullWidth
                value={value}
                onChange={(e) => onChange(`model.hyperparameters.${param.name}`, param.type === 'number' ? Number(e.target.value) : e.target.value)}
            />
        );
    };

    return(
    <Box sx={{p: 1, height: '100%', display: 'flex', flexDirection: 'column'}}>
        {/* <Typography variant="h5" gutterBottom>4. Model Training & Backtesting</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Configure your model, define backtesting parameters, and run the pipeline.
        </Typography> */}
        
        <Box sx={{ display: 'flex', flexDirection: 'row', gap: 1, height: '97.5%' }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, height: '100%', flexGrow: 0, minWidth: '400px', maxWidth: '400px' }}>
                <Paper variant="outlined" sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 1 }}>Model Selection</Typography>
                    <FormControl fullWidth size="small">
                        <InputLabel>ML Model</InputLabel>
                        <Select value={config.model.name} label="ML Model" onChange={(e) => onChange('model.name', e.target.value)}>
                            {models.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                        </Select>
                    </FormControl>
                    <Divider sx={{ pt: 2, pb: 2 }}><Typography variant="caption">Hyperparameters</Typography></Divider>
                    <Stack spacing={2} sx={{ overflowY: 'auto', pr: 1, pt: 1, flexGrow: 1 }}>
                        {modelParams.length > 0 ? (
                            modelParams.map(renderHyperparameterInput)
                        ) : (
                            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
                                No hyperparameters configured for this model.
                            </Typography>
                        )}
                    </Stack>
                    <Box sx={{ pt: 2 }}>
                        <Button variant="contained" color="primary" size="large" fullWidth onClick={onRun} disabled={isRunning}>
                            {isRunning ? <CircularProgress size={24} /> : 'Run ML Pipeline'}
                        </Button>
                    </Box>
                </Paper>

                <Paper variant="outlined" sx={{ p:2, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold', pb: 1 }}>Backtest Settings</Typography>
                    <Stack spacing={2} direction="column" mt={2} sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', flexDirection: 'row', gap: 1 }}>
                            <TextField label="Initial Capital" type="number" size="small" value={config.backtestSettings.capital} onChange={e => onChange('backtestSettings.capital', Number(e.target.value))} />
                            <TextField label="Risk per Trade (%)" type="number" size="small" value={config.backtestSettings.risk} onChange={e => onChange('backtestSettings.risk', Number(e.target.value))} />
                        </Box>
                        <Box sx={{ display: 'flex', flexDirection: 'row', gap: 1 }}>
                            <TextField label="Commission (bps)" helperText="Basis points, e.g., 2.5" type="number" size="small" value={config.backtestSettings.commissionBps} onChange={e => onChange('backtestSettings.commissionBps', Number(e.target.value))} />
                            <TextField label="Slippage (bps)" helperText="Basis points, e.g., 1.0" type="number" size="small" value={config.backtestSettings.slippageBps} onChange={e => onChange('backtestSettings.slippageBps', Number(e.target.value))} />
                        </Box>
                        <FormControlLabel control={<Switch checked={config.backtestSettings.tradeOnClose} onChange={e => onChange('backtestSettings.tradeOnClose', e.target.checked)} />} label="Trade on candle close" />
                    </Stack>
                    <Box sx={{ pt: 2 }}>
                        <Button variant="contained" color="primary" size="large" fullWidth onClick={onRun} disabled={isRunning}>
                            {isRunning ? <CircularProgress size={24} /> : 'Run Backtest'}
                        </Button>
                    </Box>
                </Paper>
            </Box>

            <Box sx={{ flexGrow: 1 }}>
                <Paper variant='outlined' sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignContent: 'center', textAlign: 'center' }}>
                    {isMLResult ? (
                        <Box sx={{ display: 'flex', flexDirection: 'row', gap: 2, height: '100%' }}>
                            <Box sx={{ width: '50%', minHeight: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', p: 2 }}>
                                <FeatureImportanceChart data={result.model_analysis.feature_importance} />
                            </Box>

                            <Box sx={{ width: '50%', display: 'flex', flexDirection: 'column', gap: 2, height: '100%', justifyContent: 'space-between', pr: 2 }}>
                                <ClassificationReportTable report={result.model_analysis.classification_report} />
                                <ConfusionMatrix
                                    matrix={result.model_analysis.confusion_matrix}
                                    labels={Object.keys(result.model_analysis.classification_report).filter(k => k.startsWith('class_'))}
                                />
                            </Box>
                        </Box>
                    ) : (
                        <Typography variant="body1" color="text.secondary">
                            Run the ML pipeline to see the analysis results.
                        </Typography>
                    )}
                </Paper>
            </Box>
        </Box>
    </Box>
);
}