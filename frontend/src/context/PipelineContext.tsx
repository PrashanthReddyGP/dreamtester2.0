// PipelineContext.tsx
import React, {
    createContext,
    useState,
    useCallback,
    useContext,
    useRef,
    useEffect
} from 'react';
import type { ReactNode } from 'react';
import {
    applyNodeChanges,
    applyEdgeChanges,
    addEdge,
} from 'reactflow';

import type {
    Node,
    Edge,
    NodeChange,
    EdgeChange,
    Connection,
} from 'reactflow';
import { v4 as uuidv4 } from 'uuid';
import { fetchAvailableSymbols } from '../../src/services/api';
import { HYPERPARAMETER_CONFIG } from '../../src/components/pipeline/mlModels';
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

// --- Type Definitions (from PipelineEditor) ---
interface IndicatorParamDef {
    name: string;
    displayName: string;
    type: 'number' | 'string' | 'boolean';
    defaultValue: number | string | boolean;
    options?: string[];
}
interface IndicatorDefinition {
    name: string;
    params: IndicatorParamDef[];
}
type IndicatorSchema = { [key: string]: IndicatorDefinition };


// --- INITIAL STATE (Copied exactly from your original file) ---
const initialNodes: Node[] = [
    {
        id: '1',
        type: 'dataSource',
        data: {
            label: 'Data Source',
            symbol: 'ADAUSDT',
            timeframe: '15m',
            startDate: '2000-01-01',
            endDate: '2100-12-31',
        },
        position: { x: 50, y: 100 },
    }
];

const initialEdges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e2-3', source: '2', target: '3' },
];


// --- CONTEXT TYPE DEFINITION ---
interface PipelineContextType {
    nodes: Node[];
    edges: Edge[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
    onNodesChange: (changes: NodeChange[]) => void;
    onEdgesChange: (changes: EdgeChange[]) => void;
    onConnect: (connection: Connection) => void;
    reactFlowWrapperRef: React.RefObject<HTMLDivElement>;
    addNode: (nodeType: string, position: { x: number; y: number }, connectingNode?: { id: string; handleType: 'source' | 'target' }) => void;
    deleteElement: (id: string, type: 'node' | 'edge') => void;
    updateNodeData: (nodeId: string, data: any) => void;

    // Add fetched data and its state to the context type
    symbolList: string[];
    isFetchingSymbols: boolean;
    indicatorSchema: IndicatorSchema;
    isLoadingSchema: boolean;
    fetchError: string | null;

    config: MLConfig;
    workflowId: string | null;
    displayData: { data: any[], info: any | null};
    isFetching: boolean;
    isCalculating: boolean;
    isEngineering: boolean;
    isValidating: boolean;
    isRunning: boolean;
    validationInfo: any | null;
    handleConfigChange: (path: string, value: any) => void;
    handleFetchData: () => Promise<void>;
    handleCalculateFeatures: () => Promise<void>;
    handleFeatureEngineering: () => Promise<void>;
    handleValidation: () => Promise<void>;
    handleRunPipeline: () => Promise<void>;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

export const PipelineContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [nodes, setNodes] = useState<Node[]>(initialNodes);
    const [edges, setEdges] = useState<Edge[]>(initialEdges);
    const reactFlowWrapperRef = useRef<HTMLDivElement>(null);

    // --- State for fetched data (symbols, indicators) ---
    const [symbolList, setSymbolList] = useState<string[]>([]);
    const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
    const [indicatorSchema, setIndicatorSchema] = useState<IndicatorSchema>({});
    const [isLoadingSchema, setIsLoadingSchema] = useState(true);
    const [fetchError, setFetchError] = useState<string | null>(null);

    // --- Fetching Logic (moved here from PipelineEditor) ---
    useEffect(() => {
        const loadSymbols = async () => {
            setIsFetchingSymbols(true);
            setFetchError(null);
            try {
                const fetchedSymbols = await fetchAvailableSymbols('binance');
                setSymbolList(fetchedSymbols);
            } catch (err: any) {
                console.error("Failed to fetch symbols:", err);
                setFetchError(err.message || 'An unknown error occurred while fetching symbols.');
            } finally {
                setIsFetchingSymbols(false);
            }
        };
        loadSymbols();
    }, []); // Empty dependency array ensures this runs only once

    useEffect(() => {
        const fetchSchema = async () => {
            setIsLoadingSchema(true);
            try {
                const response = await fetch(`${API_URL}/api/ml/indicators`);
                if (!response.ok) throw new Error('Failed to fetch indicator schema');
                const data = await response.json();
                setIndicatorSchema(data);
            } catch (error) {
                console.error("Error fetching indicator schema:", error);
                setFetchError("Could not load indicator schema.");
            } finally {
                setIsLoadingSchema(false);
            }
        };
        fetchSchema();
    }, []); // Empty dependency array ensures this runs only once

    // --- Node Manipulation Functions ---
    const updateNodeData = useCallback((nodeId: string, data: any) => {
        setNodes((nds) =>
            nds.map((node) =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, ...data } }
                    : node
            )
        );
    }, [setNodes]);

    // --- Core Callbacks ---
    const onNodesChange = useCallback((changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)), [setNodes]);

    const onEdgesChange = useCallback((changes: EdgeChange[]) => {
        // Find which edges are being removed *before* applying the state change
        const removedEdges = changes
            .filter((c): c is { id: string; type: 'remove' } => c.type === 'remove')
            .map(c => edges.find(e => e.id === c.id))
            .filter((e): e is Edge => e !== undefined); // Ensure we only have valid Edge objects

        // Apply the changes to the edges state as usual
        setEdges((eds) => applyEdgeChanges(changes, eds));

        // Now, process the logic for the removed edges
        removedEdges.forEach(removedEdge => {
            const sourceNode = nodes.find(n => n.id === removedEdge.source);
            const targetNode = nodes.find(n => n.id === removedEdge.target);

            // If a feature node was disconnected from a processIndicators node
            if (sourceNode?.type === 'feature' && targetNode?.type === 'processIndicators') {
                const currentSelections = targetNode.data.selectedIndicators || {};
                
                // Create a new object without the key of the disconnected source node
                const { [sourceNode.id]: _, ...remainingSelections } = currentSelections;
                
                // Update the node data to uncheck the box
                updateNodeData(targetNode.id, { selectedIndicators: remainingSelections });
            }
        });
    }, [edges, nodes, setEdges, updateNodeData]); // Add dependencies
    
    const onConnect = useCallback((connection: Connection) => {
        // Add the new edge to the state
        setEdges((eds) => addEdge({ ...connection, animated: true }, eds));

        // Now, check the types of nodes being connected
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        // If a feature node is connected to a processIndicators node
        if (sourceNode?.type === 'feature' && targetNode?.type === 'processIndicators') {
            // Update the target node's data to check the box for the source node
            const updatedSelection = {
                ...(targetNode.data.selectedIndicators || {}),
                [sourceNode.id]: true, // Set to true by default
            };
            updateNodeData(targetNode.id, { selectedIndicators: updatedSelection });
        }
    }, [nodes, setEdges, updateNodeData]); // Add dependencies

    const addNode = useCallback((nodeType: string, position: { x: number, y: number }, connectingNode?: { id: string, handleType: 'source' | 'target' }) => {
        let newNodeData: any;

        // Define default code for the new node.
        const defaultCode = `def process(data):
    # 'data' is a pandas DataFrame from the previous node.
    # Your custom logic here.
    
    print("Custom node received data with shape:", data.shape)
    
    # Example: Add a new column
    # data['new_col'] = 1 
    
    return data
`;

        switch (nodeType) {
            case 'feature':
            newNodeData = { label: 'Feature', feature: 'SMA', length: 20 };

                break;
            case 'dataSource':
                newNodeData = { 
                    label: 'Data Source', 
                    symbol: 'ADAUSDT', 
                    timeframe: '15m', 
                    startDate: '2000-01-01',
                    endDate: '2100-12-31',
                };
                break;
            case 'processIndicators':
                newNodeData = {
                    label: 'Process Indicators',
                    selectedIndicators: {}, // Initialize with an empty object
                };
                break;
            case 'customCode':
                newNodeData = {
                    label: 'Custom Code',
                    code: defaultCode,
                };
                break;
            case 'mlModel': { // Use block scope to declare constants
                const defaultModelName = 'RandomForestClassifier';
                const defaultConfig = HYPERPARAMETER_CONFIG[defaultModelName];
                const defaultHyperparameters = defaultConfig.reduce((acc, param) => {
                    acc[param.name] = param.defaultValue;
                    return acc;
                }, {} as { [key: string]: any });
                
                newNodeData = {
                    label: 'ML Model',
                    modelName: defaultModelName,
                    hyperparameters: defaultHyperparameters,
                };
                break;
            };
            default:
                newNodeData = { label: `New ${nodeType} Node` };
                break;
        }

        const newNode: Node = {
            id: uuidv4(),
            type: nodeType,
            position,
            data: newNodeData,
        };

        setNodes((nds) => [...nds, newNode]);

        if (connectingNode) {
            const newEdge: Edge = {
                id: uuidv4(),
                source: connectingNode.handleType === 'source' ? connectingNode.id : newNode.id,
                target: connectingNode.handleType === 'source' ? newNode.id : connectingNode.id,
            };
            
            // Manually call onConnect logic for edges created programmatically
            onConnect(newEdge);
        }
    }, [setNodes, setEdges, onConnect]); // IMPORTANT: Add onConnect here


    const deleteElement = useCallback((id: string, type: 'node' | 'edge') => {
        if (type === 'node') {
            setEdges((eds) => eds.filter((edge) => edge.source !== id && edge.target !== id));
            setNodes((nds) => nds.filter((node) => node.id !== id));
        } else if (type === 'edge') {
            // Manually trigger the removal logic when deleting an edge directly
            onEdgesChange([{ id, type: 'remove' }]);
        }
    }, [setNodes, setEdges, onEdgesChange]); // IMPORTANT: Add onEdgesChange here




    const [config, setConfig] = useState<MLConfig>(initialConfig);
    const [workflowId, setWorkflowId] = useState<string | null>(null);
    const [displayData, setDisplayData] = useState<{ data: any[], info: any | null }>({ data: [], info: null });
    const [isFetching, setIsFetching] = useState(false);
    const [isCalculating, setIsCalculating] = useState(false);
    const [isEngineering, setIsEngineering] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [validationInfo, setValidationInfo] = useState<any | null>(null);

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




    const value = {
        nodes,
        edges,
        setNodes,
        setEdges,
        onNodesChange,
        onEdgesChange,
        onConnect,
        reactFlowWrapperRef,
        addNode,
        deleteElement,
        updateNodeData,
        symbolList,
        isFetchingSymbols,
        indicatorSchema,
        isLoadingSchema,
        fetchError,
        config,
        workflowId,
        displayData,
        isFetching,
        isCalculating,
        isEngineering,
        isValidating,
        isRunning,
        validationInfo,
        handleConfigChange,
        handleFetchData,
        handleCalculateFeatures,
        handleFeatureEngineering,
        handleValidation,
        handleRunPipeline,
    };

    return (
        <PipelineContext.Provider value={value}>
            {children}
        </PipelineContext.Provider>
    );
};

export const usePipeline = (): PipelineContextType => {
    const context = useContext(PipelineContext);
    if (context === undefined) {
        throw new Error('usePipeline must be used within a PipelineContextProvider');
    }
    return context;
};