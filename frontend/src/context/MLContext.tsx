import React, { createContext, useState, useEffect, useContext } from 'react';
import type { ReactNode } from 'react';
import type { MLConfig } from '../components/machinelearning/types'; // Adjust path if needed
import type { LabelingTemplate } from '../components/machinelearning/LabelingTemplates';
import type { FETemplate } from '../components/machinelearning/FeatureEngineeringTemplates';
import { INITIAL_LABELING_TEMPLATES } from '../components/machinelearning/LabelingTemplates';
import { INITIAL_FEATURE_ENGINEERING_CODE } from '../components/machinelearning/FeatureEngineeringTemplates';
import { useTerminal } from './TerminalContext'; // We'll need this for running the pipeline

// --- Constants and Initial State (Copied from MachineLearning.tsx) ---
const API_URL = 'http://127.0.0.1:8000';

const initialConfig: MLConfig = {
    problemDefinition: { type: 'template', templateKey: 'triple_barrier', customCode: INITIAL_LABELING_TEMPLATES.triple_barrier.code },
    dataSource: { symbol: 'ADAUSDT', timeframe: '15m', startDate: '2000-01-01', endDate: '2100-12-31' },
    features: [],
    model: { name: 'RandomForestClassifier', hyperparameters: {n_estimators: 100, max_depth: 10, min_samples_split: 2, min_samples_leaf: 1, class_weight: 'balanced'}, },
    validation: { method: 'train_test_split', trainSplit: 70, walkForwardTrainWindow: 365, walkForwardTestWindow: 30 },
    preprocessing: { scaler: 'none', removeCorrelated: false, correlationThreshold: 0.9, usePCA: false, pcaComponents: 5, featureType: 'template', featureTemplateKey: 'none', customFeatureCode: INITIAL_FEATURE_ENGINEERING_CODE.guide.code },
    backtestSettings: { capital: 1000, risk: 1, commissionBps: 2.5, slippageBps: 1.0, tradeOnClose: true },
};

// --- Define the shape of our context ---
interface MLContextValue {
    // State
    config: MLConfig;
    workflowId: string | null;
    displayData: { data: any[], info: any | null };
    isFetching: boolean;
    isCalculating: boolean;
    isEngineering: boolean;
    isValidating: boolean;
    isRunning: boolean;
    validationInfo: any | null;
    labelingTemplates: { [key: string]: LabelingTemplate };
    feTemplates: { [key: string]: FETemplate };

    // State Setters & Handlers
    handleConfigChange: (path: string, value: any) => void;
    handleFetchData: () => Promise<void>;
    handleCalculateFeatures: () => Promise<void>;
    handleFeatureEngineering: () => Promise<void>;
    handleValidation: () => Promise<void>;
    handleRunPipeline: () => Promise<void>;

    // Template Management
    saveLabelingTemplate: (name: string) => Promise<void>;
    deleteLabelingTemplate: (key: string) => Promise<void>;
    saveFeTemplate: (name: string) => Promise<void>;
    deleteFeTemplate: (key: string) => Promise<void>;
    setLabelingTemplates: React.Dispatch<React.SetStateAction<{ [key: string]: LabelingTemplate; }>>;
    setFeTemplates: React.Dispatch<React.SetStateAction<{ [key: string]: FETemplate; }>>;

    resetLoadingStates: () => void;
}

const MLContext = createContext<MLContextValue | undefined>(undefined);

// --- Create the Provider Component ---
export const MLProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    // All state from MachineLearning.tsx is now here
    const [config, setConfig] = useState<MLConfig>(initialConfig);
    const [workflowId, setWorkflowId] = useState<string | null>(null);
    const [displayData, setDisplayData] = useState<{ data: any[], info: any | null }>({ data: [], info: null });
    const [isFetching, setIsFetching] = useState(false);
    const [isCalculating, setIsCalculating] = useState(false);
    const [isEngineering, setIsEngineering] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [validationInfo, setValidationInfo] = useState<any | null>(null);
    const [labelingTemplates, setLabelingTemplates] = useState<{ [key: string]: LabelingTemplate }>(INITIAL_LABELING_TEMPLATES);
    const [feTemplates, setFeTemplates] = useState<{ [key: string]: FETemplate }>(INITIAL_FEATURE_ENGINEERING_CODE);
    const [deleteTemplateState, setDeleteTemplateState] = useState<{ open: boolean; key: string | null }>({ open: false, key: null });
    const [deleteFeState, setDeleteFeState] = useState<{ open: boolean; key: string | null }>({ open: false, key: null });

    const { connectToBatch, toggleTerminal } = useTerminal();

    // All effects are also moved here
    useEffect(() => {
        const startNewWorkflow = async () => {
            try {
                const response = await fetch(`${API_URL}/api/ml/workflow/start`);
                const data = await response.json();
                if (data.workflow_id) {
                    setWorkflowId(data.workflow_id);
                    console.log("Started ML Workflow with ID:", data.workflow_id);
                }
            } catch (error) {
                console.error("Failed to start ML workflow:", error);
            }
        };
        // Only start a workflow if we don't already have one
        if (!workflowId) {
            startNewWorkflow();
        }
    }, [workflowId]); // Depend on workflowId to prevent re-running

    useEffect(() => {
        const fetchAllCustomTemplates = async () => {
            try {
                const response = await fetch(`${API_URL}/api/ml/templates`);
                if (!response.ok) throw new Error("Failed to fetch labeling templates");
                const customTemplates = await response.json();
                setLabelingTemplates(prev => ({ ...INITIAL_LABELING_TEMPLATES, ...customTemplates }));
            } catch (error) { console.error("Could not load labeling templates:", error); }

            try {
                const response = await fetch(`${API_URL}/api/ml/fe-templates`);
                if (!response.ok) throw new Error("Failed to fetch FE templates");
                const customTemplates = await response.json();
                setFeTemplates(prev => ({ ...INITIAL_FEATURE_ENGINEERING_CODE, ...customTemplates }));
            } catch (error) { console.error("Could not load FE templates:", error); }
        };
        fetchAllCustomTemplates();
    }, []);

    // Other effects for syncing editors
    useEffect(() => {
        if (config.problemDefinition.type === 'template') {
            const newCode = labelingTemplates[config.problemDefinition.templateKey]?.code || '';
            if (newCode !== config.problemDefinition.customCode) {
                handleConfigChange('problemDefinition.customCode', newCode);
            }
        }
    }, [config.problemDefinition.type, config.problemDefinition.templateKey, labelingTemplates]);

    useEffect(() => {
        if (config.preprocessing.featureType === 'template') {
            const newCode = feTemplates[config.preprocessing.featureTemplateKey]?.code || '';
            if (newCode !== config.preprocessing.customFeatureCode) {
                handleConfigChange('preprocessing.customFeatureCode', newCode);
            }
        }
    }, [config.preprocessing.featureType, config.preprocessing.featureTemplateKey, feTemplates]);

    useEffect(() => {
        handleConfigChange('model.hyperparameters', {});
    }, [config.model.name]);


    const handleConfigChange = (path: string, value: any) => {
        setConfig(prev => {
            const keys = path.split('.');
            let tempState = { ...prev };
            let current = tempState as any;
            for (let i = 0; i < keys.length - 1; i++) {
                current[keys[i]] = { ...current[keys[i]] };
                current = current[keys[i]];
            }
            current[keys[keys.length - 1]] = value;
            return tempState;
        });
    };

    // --- Template Management Logic ---
    const saveLabelingTemplate = async (templateName: string) => {
        const templateKey = `custom_${templateName.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
        
        const newTemplate: LabelingTemplate = {
            name: templateName,
            description: 'A custom user-defined template.',
            code: config.problemDefinition.customCode,
            isDeletable: true,
        };

        try {
            // --- API Call to save template ---
            await fetch(`${API_URL}/api/ml/templates`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: templateKey, ...newTemplate }),
            });

            // Update local state on success
            const updatedTemplates = { ...labelingTemplates, [templateKey]: newTemplate };
            setLabelingTemplates(updatedTemplates);

            setConfig(prev => ({
                ...prev,
                problemDefinition: {
                    ...prev.problemDefinition,
                    type: 'template',
                    templateKey: templateKey,
                }
            }));
        } catch (error) {
            console.error("Failed to save template:", error);
            alert("Error: Could not save the template to the server.");
        }
    };

    const deleteLabelingTemplate = async () => {
        if (!deleteTemplateState.key) return;

        try {
            // --- API Call to delete template ---
            await fetch(`${API_URL}/api/ml/templates/${deleteTemplateState.key}`, {
                method: 'DELETE',
            });
            
            // Update local state on success
            const { [deleteTemplateState.key]: _, ...remainingTemplates } = labelingTemplates;
            setLabelingTemplates(remainingTemplates);

            if (config.problemDefinition.templateKey === deleteTemplateState.key) {
                const firstTemplateKey = Object.keys(remainingTemplates)[0];
                setConfig(prev => ({
                    ...prev,
                    problemDefinition: { ...prev.problemDefinition, templateKey: firstTemplateKey }
                }));
            }
        } catch (error) {
            console.error("Failed to delete template:", error);
            alert("Error: Could not delete the template from the server.");
        } finally {
            setDeleteTemplateState({ open: false, key: null });
        }
    };

    // --- Template Management for FEATURE ENGINEERING ---
    const saveFeTemplate = async (templateName: string) => {
        const templateKey = `custom_${templateName.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
        const newTemplate: FETemplate = {
            name: templateName,
            description: 'A custom feature engineering process.',
            code: config.preprocessing.customFeatureCode,
            isDeletable: true,
        };

        try {
            await fetch(`${API_URL}/api/ml/fe-templates`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: templateKey, ...newTemplate }),
            });
            setFeTemplates(prev => ({ ...prev, [templateKey]: newTemplate }));
            handleConfigChange('preprocessing.featureTemplateKey', templateKey);
            handleConfigChange('preprocessing.featureType', 'template');
        } catch (error) { console.error("Failed to save FE template:", error); }
    };

    const deleteFeTemplate = async () => {
        if (!deleteFeState.key) return;
        try {
            await fetch(`${API_URL}/api/ml/fe-templates/${deleteFeState.key}`, { method: 'DELETE' });
            
            const { [deleteFeState.key]: _, ...remaining } = feTemplates;
            setFeTemplates(remaining);

            if (config.preprocessing.featureTemplateKey === deleteFeState.key) {
                handleConfigChange('preprocessing.featureTemplateKey', Object.keys(remaining)[0]);
            }
        } catch (error) { console.error("Failed to delete FE template:", error); } 
        finally { setDeleteFeState({ open: false, key: null }); }
    };

    const handleFetchData = async () => {
        if (!workflowId) return alert("Workflow not initialized. Please wait or refresh.");
        
        setIsFetching(true);
        // Clear previous data and info immediately for better user feedback
        setDisplayData({ data: [], info: null });

        try {
            // NOTE: The backend endpoint should be designed to return data directly, not a batch_id.
            // Let's assume the endpoint is `/api/ml/fetch-data`.
            const response = await fetch(`${API_URL}/api/ml/workflow/${workflowId}/fetch-data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // It's good practice to send only the data the endpoint needs.
                body: JSON.stringify(config.dataSource),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred while fetching data.' }));
                throw new Error(errorData.detail);
            }

            // Expect the backend to return a structure like: { data: [...], info: {...} }
            const result = await response.json();

            if (result.data && result.info) {
                // Update the state with the fetched data, which will re-render the child components
                setDisplayData({ data: result.data, info: result.info });
            } else {
                throw new Error("Received an invalid data structure from the server.");
            }

        } catch (error) {
            // Handle any errors during the fetch
            console.error("Failed to fetch OHLCV data:", error);
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
            alert(`Error fetching data: ${errorMessage}`);
            // You can also set an error message in the info panel
            setDisplayData({ data: [], info: { error: errorMessage } });
        } finally {
            // CRUCIAL: Always set the loading state to false in the finally block
            setIsFetching(false);
        }
    };

    const handleCalculateFeatures = async () => {
        if (!workflowId) return alert("Workflow not initialized. Please wait or refresh.");
        
        setIsCalculating(true);

        try {
            // The payload must include the data source, the list of features, and preprocessing steps.
            const payload = {
                dataSource: config.dataSource,
                features: config.features,
            };

            const response = await fetch(`${API_URL}/api/ml/workflow/${workflowId}/calculate-features`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload), // Send the complete payload
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Server error during feature calculation.' }));
                throw new Error(errorData.detail);
            }

            const result = await response.json();
            if (result.data && result.info) {
                setDisplayData({ data: result.data, info: result.info });
            } else {
                throw new Error("Invalid data structure received for features.");
            }

        } catch (error) {
            console.error("Failed to calculate features:", error);
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
            alert(`Error calculating features: ${errorMessage}`);
            setDisplayData({ data: [], info: { error: errorMessage } });
        } finally {
            setIsCalculating(false);
        }
    };

    const handleFeatureEngineering = async () => {
        if (!workflowId) return alert("Workflow not initialized. Please wait or refresh.");

        setIsEngineering(true);

        try {
            // The payload must include the data source, the list of features, and preprocessing steps.
            const payload = {
                dataSource: config.dataSource,
                features: config.features,
                preprocessing: config.preprocessing
            };

            const response = await fetch(`${API_URL}/api/ml/workflow/${workflowId}/feature-engineering`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload), // Send the complete payload
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Server error during feature engineering.' }));
                throw new Error(errorData.detail);
            }

            const result = await response.json();
            if (result.data && result.info) {
                setDisplayData({ data: result.data, info: result.info });
            } else {
                throw new Error("Invalid data structure received for features.");
            }

        } catch (error) {
            console.error("Failed to feature engineer:", error);
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
            alert(`Error feature engineering: ${errorMessage}`);
            setDisplayData({ data: [], info: { error: errorMessage } });
        } finally {
            setIsEngineering(false);
        }
    };


    const handleValidation = async () => {
        if (!workflowId) return alert("Workflow not initialized. Please wait or refresh.");

        setIsValidating(true);
        setValidationInfo(null); // Clear previous results

        try {
            // The payload must include the data source, the list of features, and preprocessing steps.
            const payload = {
                dataSource: config.dataSource,
                features: config.features,
                preprocessing: config.preprocessing,
                validation: config.validation,
                problemDefinition: config.problemDefinition
            };

            const response = await fetch(`${API_URL}/api/ml/workflow/${workflowId}/data-validation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload), // Send the complete payload
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Server error during data validation.' }));
                throw new Error(errorData.detail);
            }

            const result = await response.json();

            if (result.data && result.info) {
                // We update both the main display and the specific validation info
                setDisplayData({ data: result.data, info: result.info });
                if (result.info.validation_info) {
                    setValidationInfo(result.info.validation_info);
                }
            } else {
                throw new Error("Invalid data structure received for features.");
            }

        } catch (error) {
            console.error("Failed to validate data:", error);
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
            alert(`Error validating data: ${errorMessage}`);
            setValidationInfo({ error: errorMessage });
        } finally {
            setIsValidating(false);
        }
    };

    const handleRunPipeline = async () => {
        if (!workflowId) return alert("Workflow not initialized. Please wait or refresh.");

        setIsRunning(true);
        toggleTerminal(true);
        console.log("Submitting ML Pipeline with config:", JSON.stringify(config, null, 2));
        
        try {
            // --- CALL THE NEW, STATEFUL ENDPOINT ---
            const response = await fetch(`${API_URL}/api/ml/workflow/${workflowId}/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config), // Send the final config
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
            }

            const result: { batch_id: string } = await response.json();
            
            if (result.batch_id) {
                connectToBatch(result.batch_id);
                // navigate('/analysis');
            } else {
                throw new Error("Submission was successful, but no batch ID was returned.");
            }

        } catch (error) {
            console.error("Failed to run ML Pipeline:", error);
            alert(`Error submitting pipeline: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
            setIsRunning(false);
        } finally {
            setIsRunning(false)
        }
    };
    
    const resetLoadingStates = () => {
        console.log("Resetting all ML loading states.");
        setIsFetching(false);
        setIsCalculating(false);
        setIsEngineering(false);
        setIsValidating(false);
        setIsRunning(false);
    };

    // The value provided to consuming components
    const value: MLContextValue = {
        config,
        workflowId,
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
        setLabelingTemplates,
        setFeTemplates,
        resetLoadingStates
    };

    return <MLContext.Provider value={value}>{children}</MLContext.Provider>;
};

// --- Create a custom hook for easy consumption ---
export const useML = (): MLContextValue => {
    const context = useContext(MLContext);
    if (context === undefined) {
        throw new Error('useML must be used within a MLProvider');
    }
    return context;
};