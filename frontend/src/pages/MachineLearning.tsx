import React, { useState, useEffect } from 'react';
import { Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';

// Import types and child components (keep these)
import type { MLConfig, StrategyResult, MLResult } from '../services/api'; // Adjust imports as needed
import { MlSidebar } from '../components/machinelearning/MLSidebar';
import { DataTab } from '../components/machinelearning/tabs/DataTab';
import { FeatureEngineeringTab } from '../components/machinelearning/tabs/FeatureEngineeringTab';
import { LabelingDataSplitTab } from '../components/machinelearning/tabs/LabelingDataSplitTab';
import { TrainingBacktestingTab } from '../components/machinelearning/tabs/TrainingBacktestingTab';
import { NameInputDialog } from '../components/common/NameItemDialog';
import { ConfirmationDialog } from '../components/common/ConfirmationDialog';
import { useAnalysis } from '../context/AnalysisContext';
import { useML } from '../context/MLContext'; // <-- IMPORT OUR NEW HOOK

// --- Supervised Learning: Classification Models ---
// For predicting a category (e.g., "Buy", "Sell", "Hold")
const CLASSIFICATION_MODELS = [
    'LogisticRegression',
    'RandomForestClassifier',
    'LightGBMClassifier',
    'XGBoostClassifier',
    'SVC', // Support Vector Classifier
    'KNeighborsClassifier',
    'DecisionTreeClassifier',
    'GaussianNB', // Gaussian Naive Bayes
];

// --- Supervised Learning: Regression Models ---
// For predicting a continuous value (e.g., future price, volatility)
const REGRESSION_MODELS = [
    'LinearRegression',
    'RandomForestRegressor',
    'LightGBMRegressor',
    'XGBoostRegressor',
    'SVR', // Support Vector Regressor
    'Ridge', // Ridge Regression
    'Lasso', // Lasso Regression
];

// --- Unsupervised Learning Models ---
// For discovering hidden patterns, groups, or anomalies
const UNSUPERVISED_MODELS = [
    // Clustering
    'KMeans',
    'DBSCAN',
    'AgglomerativeClustering',

    // Dimensionality Reduction
    'PCA', // Principal Component Analysis

    // Anomaly Detection
    'IsolationForest',
    'OneClassSVM',
];

// --- A Combined List for a Single Dropdown ---
const ALL_MOCK_MODELS = [
    ...CLASSIFICATION_MODELS,
    ...REGRESSION_MODELS,
    ...UNSUPERVISED_MODELS,
];

const MODELS = ALL_MOCK_MODELS;
type AnalysisResult = StrategyResult | MLResult;

export const MachineLearning: React.FC = () => {
    // --- ALL STATE AND LOGIC IS NOW GONE ---
    // Instead, we consume everything from our context.
    const {
        config,
        displayData,
        isFetching,
        isCalculating,
        isEngineering,
        isValidating,
        isRunning,
        validationInfo,
        labelingTemplates,
        feTemplates,
        handleConfigChange,
        handleFetchData,
        handleCalculateFeatures,
        handleFeatureEngineering,
        handleValidation,
        handleRunPipeline,
        saveLabelingTemplate,
        deleteLabelingTemplate,
        saveFeTemplate,
        deleteFeTemplate,
        resetLoadingStates
    } = useML();

    // Local UI state (like which tab is active or dialogs) is fine to keep here.
    const [activeTab, setActiveTab] = useState<'data' | 'features' | 'labeling' | 'training'>('data');
    const [isSaveTemplateDialogOpen, setIsSaveTemplateDialogOpen] = useState(false);
    const [deleteTemplateState, setDeleteTemplateState] = useState<{ open: boolean; key: string | null }>({ open: false, key: null });
    const [isSaveFeDialogOpen, setIsSaveFeDialogOpen] = useState(false);
    const [deleteFeState, setDeleteFeState] = useState<{ open: boolean; key: string | null }>({ open: false, key: null });
    
    // Hooks that don't manage the core workflow state can also stay.
    const navigate = useNavigate();
    const { results } = useAnalysis();
    const [selectedStrategy, setSelectedStrategy] = React.useState<AnalysisResult | null>(null);

    useEffect(() => {
        if (results.length > 0) {
            if (!selectedStrategy || !results.some(r => r.strategy_name === selectedStrategy.strategy_name)) {
                setSelectedStrategy(results[0]);
            }
        } else {
            setSelectedStrategy(null);
        }
    }, [results, selectedStrategy]);

    // --- NEW: Add this useEffect for cleanup ---
    useEffect(() => {
        // This function is the cleanup function.
        // It will run when the component is about to unmount.
        return () => {
            resetLoadingStates();
        };
    }, []); // The empty dependency array means this effect runs only on mount and unmount.

    // Handler functions now just call the context functions
    const handleSaveTemplate = async (templateName: string) => {
        await saveLabelingTemplate(templateName);
        setIsSaveTemplateDialogOpen(false);
    };

    const handleDeleteTemplateConfirm = async () => {
        if (!deleteTemplateState.key) return;
        await deleteLabelingTemplate(deleteTemplateState.key);
        setDeleteTemplateState({ open: false, key: null });
    };

    const handleSaveFeTemplate = async (templateName: string) => {
        await saveFeTemplate(templateName);
        setIsSaveFeDialogOpen(false);
    };

    const handleDeleteFeConfirm = async () => {
        if (!deleteFeState.key) return;
        await deleteFeTemplate(deleteFeState.key);
        setDeleteFeState({ open: false, key: null });
    };

    const renderActiveTab = () => {
        switch(activeTab) {
            case 'data':
                return <DataTab 
                    config={config} 
                    onChange={handleConfigChange} 
                    onFetch={handleFetchData} 
                    onCalculate={handleCalculateFeatures} 
                    displayData={displayData.data} 
                    displayInfo={displayData.info} 
                    isFetching={isFetching} 
                    isCalculating={isCalculating} 
                    />;
            case 'features':
                return <FeatureEngineeringTab 
                    config={config} 
                    onChange={handleConfigChange} 
                    onEngineer={handleFeatureEngineering} 
                    onSaveTemplate={() => setIsSaveFeDialogOpen(true)} 
                    onDeleteTemplate={(key) => setDeleteFeState({ open: true, key })} 
                    displayData={displayData.data} 
                    displayInfo={displayData.info}
                    isEngineering={isEngineering} 
                    feTemplates={feTemplates} 
                    />;
            case 'labeling':
                return <LabelingDataSplitTab 
                    config={config} 
                    onChange={handleConfigChange} 
                    labelingTemplates={labelingTemplates} 
                    onDeleteTemplate={(key) => setDeleteTemplateState({ open: true, key })} 
                    onSaveTemplate={() => setIsSaveTemplateDialogOpen(true)}
                    onValidate={handleValidation}
                    isValidating={isValidating}
                    validationInfo={validationInfo}
                    />;
            case 'training':
                return <TrainingBacktestingTab 
                    config={config} 
                    onChange={handleConfigChange} 
                    onRun={handleRunPipeline} 
                    isRunning={isRunning} 
                    models={MODELS}
                    result={selectedStrategy}
                    />;
            default:
                return null;
        }
    };

    return (
        <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)', width: '100vw' }}>
            <MlSidebar activeTab={activeTab} setActiveTab={setActiveTab} />
            <Box component="main" sx={{ flexGrow: 1, overflowY: 'auto', height: '100%' }}>
                {renderActiveTab()}
            </Box>

            <NameInputDialog open={isSaveTemplateDialogOpen} onClose={() => setIsSaveTemplateDialogOpen(false)} onConfirm={handleSaveTemplate} dialogTitle="Save as Template" confirmButtonText="Save" />
            <ConfirmationDialog open={deleteTemplateState.open} onClose={() => setDeleteTemplateState({ open: false, key: null })} onConfirm={handleDeleteTemplateConfirm} title="Delete Template?" />
            <NameInputDialog open={isSaveFeDialogOpen} onClose={() => setIsSaveFeDialogOpen(false)} onConfirm={handleSaveFeTemplate} dialogTitle="Save Feature Engineering Template" />
            <ConfirmationDialog open={deleteFeState.open} onClose={() => setDeleteFeState({ open: false, key: null })} onConfirm={handleDeleteFeConfirm} title="Delete Feature Engineering Template?" />
        </Box>
    );
};