// PipelineContext.tsx
import React, {
    createContext,
    useState,
    useCallback,
    useContext,
    useRef,
    useEffect,
    useMemo
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
import { INITIAL_LABELING_TEMPLATES } from '../components/pipeline/templates/LabelingTemplates';
import { INITIAL_FEATURE_ENGINEERING_CODE } from '../components/pipeline/templates/FeaturesEngineeringTemplates';
import { useTerminal } from './TerminalContext'; // We'll need this for running the pipeline
import type { IndicatorSchema } from '../components/pipeline/types';
import { TUNING_GRID_CONFIG } from '../components/pipeline/mlModels';

// --- Constants and Initial State (Copied from MachineLearning.tsx) ---
const API_URL = 'http://127.0.0.1:8000';



// 1. Define the new structured data for the BacktesterNode
export interface BacktesterNodeData {
    label: string;
    selectedTemplateKey?: string;
    config: {
        initialCapital: number;
        riskPercent: number;
        rr: number;
        exitType: 'tp_sl' | 'single_condition' | 'time_based';
        tradeDirection: 'long' | 'short' | 'hedge';
        // Add more config fields here in the future
    };
    codeBlocks: {
        indicators: string;
        entryLogic: string;
        exitLogic: string; // Only used if exitType is 'single_condition'
    };
}

// 2. The template will now store a stringified version of BacktesterNodeData
export interface BacktestTemplate {
    name: string;
    description: string;
    code: string; // This will be JSON.stringify(BacktesterNodeData)
    isDeletable: boolean;
}

// 3. Update the initial template to use the new structure
const defaultBacktesterData: Omit<BacktesterNodeData, 'label' | 'selectedTemplateKey'> = {
    config: {
        initialCapital: 1000,
        riskPercent: 1,
        rr: 1,
        exitType: 'tp_sl',
        tradeDirection: 'hedge',
    },
    codeBlocks: {
        indicators: `[
    ('SMA', self.timeframe, (50,)),
    ('SMA', self.timeframe, (200,)),
]`,
        entryLogic: `fast_sma = self.df[f'sma_50_{self.timeframe}'].values
slow_sma = self.df[f'sma_200_{self.timeframe}'].values

# Condition for long entry
if fast_sma[i-1] < slow_sma[i-1] and fast_sma[i] > slow_sma[i]:
    return True # Return True to signal a long entry

return False`,
        exitLogic: `# Example: Exit if price closes below the slow moving average
slow_sma = self.df[f'sma_200_{self.timeframe}'].values
current_price = self.close[i]

if current_price < slow_sma[i]:
    return True # Return True to signal an exit

return False`
    }
};

const INITIAL_BACKTEST_TEMPLATES: Record<string, BacktestTemplate> = {
    'ma_crossover': {
        name: 'MA Crossover Guide',
        description: 'A simple moving average crossover strategy example.',
        code: JSON.stringify({ // Store the structured data as a string
            ...defaultBacktesterData,
            label: 'MA Crossover Guide',
        }),
        isDeletable: false,
    }
};




export interface FETemplate {
    name: string;
    description: string;
    code: string;
    isDeletable: boolean;
}

export interface LabelingTemplate {
    name: string;
    description: string;
    code: string;
    isDeletable: boolean;
}

// --- Type Definitions (from PipelineEditor) ---
interface PipelineNodeData {
  data: any[];
  info: any | null;
}

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

// The initial pipeline has only one node, so it cannot have any edges.
const initialEdges: Edge[] = []; 

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
    addNode: (nodeType: string, position: { x: number; y: number }, connectingNode?: { id: string; handleType: 'source' | 'target' }, template?: { label: string; code: string; subType: 'feature_engineering' | 'labeling' }) => void;
    deleteElement: (id: string, type: 'node' | 'edge') => void;
    updateNodeData: (nodeId: string, data: any) => void;
    feTemplates: Record<string, FETemplate>;
    labelingTemplates: Record<string, LabelingTemplate>;
    backtestTemplates: Record<string, BacktestTemplate>;
    saveFeTemplate: (name: string, description: string, code: string) => Promise<void>;
    saveLabelingTemplate: (name: string, description: string, code: string) => Promise<void>;
    saveBacktestTemplate: (name: string, description: string, code: string) => Promise<string>;
    deleteFeTemplate: (templateKey: string) => Promise<void>;
    deleteLabelingTemplate: (templateKey: string) => Promise<void>;
    deleteBacktestTemplate: (templateKey: string) => Promise<void>;

    // Add fetched data and its state to the context type
    symbolList: string[];
    isFetchingSymbols: boolean;
    indicatorSchema: IndicatorSchema;
    isLoadingSchema: boolean;
    fetchError: string | null;

    workflowId: string | null;
    isProcessing: boolean; // A single global processing state
    processingNodeId: string | null; // Which node is currently running?
    selectedNodeId: string | null; // Which node is selected for viewing?
    pipelineNodeCache: Record<string, PipelineNodeData>; // Cache for node outputs
    selectNode: (nodeId: string | null) => void;
    executePipelineUpToNode: (nodeId: string) => Promise<void>;
    dataForDisplay: { data: any[], info: any | null};
    
    newWorkflow: () => Promise<void>;
    currentWorkflowName: string | null;
    savedWorkflows: string[];
    saveWorkflow: (name: string) => Promise<void>;
    loadWorkflow: (name: string) => Promise<void>;

    clearBackendCache: () => Promise<void>;

    editingNodeId: string | null;
    setEditingNodeId: React.Dispatch<React.SetStateAction<string | null>>;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

export const PipelineContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [nodes, setNodes] = useState<Node[]>(initialNodes);
    const [edges, setEdges] = useState<Edge[]>(initialEdges);
    const reactFlowWrapperRef = useRef<HTMLDivElement>(null);
    const [workflowId, setWorkflowId] = useState<string | null>(null);

    // --- State for fetched data (symbols, indicators) ---
    const [symbolList, setSymbolList] = useState<string[]>([]);
    const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
    const [indicatorSchema, setIndicatorSchema] = useState<IndicatorSchema>({});
    const [isLoadingSchema, setIsLoadingSchema] = useState(true);
    const [fetchError, setFetchError] = useState<string | null>(null);

    const [isProcessing, setIsProcessing] = useState(false);
    const [processingNodeId, setProcessingNodeId] = useState<string | null>(null);
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [pipelineNodeCache, setPipelineNodeCache] = useState<Record<string, PipelineNodeData>>({});

    const [feTemplates, setFeTemplates] = useState<Record<string, FETemplate>>(INITIAL_FEATURE_ENGINEERING_CODE);
    const [labelingTemplates, setLabelingTemplates] = useState<Record<string, LabelingTemplate>>(INITIAL_LABELING_TEMPLATES);
    const [backtestTemplates, setBacktestTemplates] = useState<Record<string, BacktestTemplate>>(INITIAL_BACKTEST_TEMPLATES);

    const [savedWorkflows, setSavedWorkflows] = useState<string[]>([]);

    const [currentWorkflowName, setCurrentWorkflowName] = useState<string | null>(null);

    // Add the new state for tracking the editing node
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
    
    const { connectToBatch, toggleTerminal } = useTerminal();

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

    useEffect(() => {

        const fetchTemplates = async () => {
            try {
                // Fetch Feature Engineering Templates
                const feResponse = await fetch(`${API_URL}/api/ml/fe-templates`);
                const savedFeTemplates = await feResponse.json();
                // Merge initial (non-deletable) with saved (deletable)
                setFeTemplates(prev => ({ ...prev, ...savedFeTemplates }));

                // Fetch Labeling Templates
                const labelingResponse = await fetch(`${API_URL}/api/ml/templates`);
                const savedLabelingTemplates = await labelingResponse.json();
                setLabelingTemplates(prev => ({ ...prev, ...savedLabelingTemplates }));

                // 7. Fetch Backtest Templates
                const btResponse = await fetch(`${API_URL}/api/ml/backtest-templates`);
                const savedBtTemplates = await btResponse.json();
                setBacktestTemplates(prev => ({ ...prev, ...savedBtTemplates }));

            } catch (error) {
                console.error("Failed to fetch custom templates:", error);
            }
        };

        fetchTemplates();
    }, []); // This runs once on mount

    const fetchSavedWorkflows = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/pipelines/list`);
            if (!response.ok) throw new Error('Failed to fetch workflows');
            const data = await response.json();
            setSavedWorkflows(data.workflows || []);
        } catch (error) {
            console.error("Error fetching saved workflows:", error);
            setSavedWorkflows([]);
        }
    }, [])

    useEffect(() => {
        fetchSavedWorkflows();
    }, [fetchSavedWorkflows]);

// --- Functions for Save/Load ---
    const saveWorkflow = useCallback(async (name: string) => {
        try {
            // We only need to save nodes and edges for reconstruction
            const workflow = { nodes, edges };
            const response = await fetch(`${API_URL}/api/pipelines/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, workflow }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save workflow');
            }
            await fetchSavedWorkflows(); // Refresh the list of saved workflows
            setCurrentWorkflowName(name);
        } catch (error) {
            console.error("Error saving workflow:", error);
            alert(`Error: Could not save workflow. ${error instanceof Error ? error.message : ''}`);
        }
    }, [nodes, edges, fetchSavedWorkflows]);

    const loadWorkflow = useCallback(async (name: string) => {
        try {
            const response = await fetch(`${API_URL}/api/pipelines/load/${name}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to load workflow');
            }
            const workflow = await response.json();
            if (workflow.nodes && workflow.edges) {
                setCurrentWorkflowName(name);
                setNodes(workflow.nodes);
                setEdges(workflow.edges);
                // Reset runtime state
                setSelectedNodeId(null);
                setPipelineNodeCache({});
            } else {
                throw new Error("Invalid workflow data received from server");
            }
        } catch (error) {
            console.error("Error loading workflow:", error);
            alert(`Error: Could not load workflow. ${error instanceof Error ? error.message : ''}`);
        }
    }, [setNodes, setEdges]);

    // This function now handles overwriting and returns the key on success.
    const saveFeTemplate = useCallback(async (name: string, description: string, code: string): Promise<string> => {
        // Generate a predictable key from the name for easier lookups and updates
        const newKey = name.toLowerCase().replace(/\s+/g, '_');
        const newTemplate: FETemplate = { name, description, code, isDeletable: true };

        // Check if a template with this key already exists
        if (feTemplates[newKey]) {
            if (!window.confirm(`A template named "${name}" already exists. Do you want to overwrite it?`)) {
                // If the user cancels, throw an error to stop the process
                throw new Error("Save operation cancelled by user.");
            }
        }

        // API call to save/update. The backend should handle this as an "upsert".
        await fetch(`${API_URL}/api/ml/fe-templates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: newKey, ...newTemplate }),
        });

        // Update local state
        setFeTemplates(prev => ({ ...prev, [newKey]: newTemplate }));
        
        // Return the key so the calling component can use it
        return newKey;

    }, [feTemplates]);
    
    const saveLabelingTemplate = useCallback(async (name: string, description: string, code: string) => {
        const newKey = name.toLowerCase().replace(/\s+/g, '_') + `_${uuidv4().slice(0, 4)}`;
        const newTemplate: LabelingTemplate = { name, description, code, isDeletable: true };

        await fetch(`${API_URL}/api/ml/templates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: newKey, ...newTemplate }),
        });

        setLabelingTemplates(prev => ({ ...prev, [newKey]: newTemplate }));
        alert(`Template "${name}" saved successfully!`);
    }, []);

    // --- Node Manipulation Functions ---
    const updateNodeData = useCallback((nodeId: string, data: any) => {
        setNodes((nds) => {
            // Step 1: Apply the primary update to the target node
            const newNds = nds.map((node) =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, ...data } }
                    : node
            );

            // Step 2: Check if the updated node is a DataSource and if its timeframe changed.
            const sourceNode = newNds.find(n => n.id === nodeId);
            if (sourceNode?.type === 'dataSource' && 'timeframe' in data) {
                const newTimeframe = data.timeframe;
                
                // Find all direct children of this node
                const childEdges = edges.filter(e => e.source === nodeId);
                const childIds = new Set(childEdges.map(e => e.target));

                // Step 3: If so, propagate the new timeframe to any connected FeatureNode children.
                return newNds.map(node => {
                    if (childIds.has(node.id) && node.type === 'feature') {
                        // This is a child FeatureNode, update its timeframe
                        return {
                            ...node,
                            data: { ...node.data, timeframe: newTimeframe }
                        };
                    }
                    // Otherwise, return the node as is
                    return node;
                });
            }
            
            // If no propagation is needed, just return the result of the primary update
            return newNds;
        });
    }, [setNodes, edges]); // IMPORTANT: Add `edges` as a dependency

    const deleteFeTemplate = useCallback(async (templateKey: string) => {
        if (!window.confirm(`Are you sure you want to delete the template "${feTemplates[templateKey]?.name}"?`)) {
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/ml/fe-templates/${templateKey}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error('Failed to delete template from server.');
            }

            // If successful, update the local state to remove the template
            setFeTemplates(prev => {
                const newTemplates = { ...prev };
                delete newTemplates[templateKey];
                return newTemplates;
            });

        } catch (error) {
            console.error("Error deleting FE template:", error);
            alert("Could not delete the template.");
        }
    }, [feTemplates]); // Depend on feTemplates to get the name for the confirm dialog

    const deleteLabelingTemplate = useCallback(async (templateKey: string) => {
        if (!window.confirm(`Are you sure you want to delete the template "${labelingTemplates[templateKey]?.name}"?`)) {
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/api/ml/templates/${templateKey}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error('Failed to delete template from server.');
            }

            // If successful, update the local state
            setLabelingTemplates(prev => {
                const newTemplates = { ...prev };
                delete newTemplates[templateKey];
                return newTemplates;
            });

        } catch (error) {
            console.error("Error deleting Labeling template:", error);
            alert("Could not delete the template.");
        }
    }, [labelingTemplates]);
    
    const newWorkflow = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/ml/workflow/clear_cache`, {
                method: 'POST',
            });
            if (!response.ok) throw new Error("Server failed to clear cache.");
            
            setNodes(initialNodes);
            setEdges(initialEdges);
            setSelectedNodeId(null);
            setPipelineNodeCache({});
            setCurrentWorkflowName(null);
        } catch (error) {
            console.error("Failed to clear backend cache:", error);
            alert("Error: Could not clear backend cache.");
        }
    }, [setNodes, setEdges]);

    const clearBackendCache = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/ml/workflow/clear_cache`, {
                method: 'POST',
            });
            if (!response.ok) throw new Error("Server failed to clear cache.");
            const result = await response.json();
            alert(result.message); // e.g., "Backend cache cleared successfully."
        } catch (error) {
            console.error("Failed to clear backend cache:", error);
            alert("Error: Could not clear backend cache.");
        }
    }, []);

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
    
    // Validation Function
    const isValidConnection = (connection: Connection): boolean => {
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) return false;

        // RULE 1: ClassImbalanceNode can only connect to the 'train' output of a DataValidationNode.
        if (targetNode.type === 'classImbalance') {
            if (sourceNode.type !== 'dataValidation' || connection.sourceHandle !== 'train') {
                alert("Invalid Connection:\nThe Class Imbalance node can only be connected to the 'Train' output of a Data Validation node.");
                return false;
            }
        }

        // RULE 2: ModelTrainerNode's 'test' handle MUST connect to a DataValidationNode's 'test' output.
        if (targetNode.type === 'modelTrainer' && connection.targetHandle === 'test') {
            if (sourceNode.type !== 'dataValidation' || connection.sourceHandle !== 'test') {
                alert("Invalid Connection:\nThe Model Trainer's 'Test Data' input must be connected to the 'Test' output of a Data Validation node.");
                return false;
            }
        }

        // Add more rules here in the future...

        return true; // All checks passed
    };

    const onConnect = useCallback((connection: Connection) => {
        // Call the validation function first
        if (!isValidConnection(connection)) {
            return; // Abort the connection if it's invalid
        }

        // This function will now ONLY handle connections between EXISTING nodes.
        // The logic is still correct for that use case.
        setEdges((eds) => addEdge({ ...connection, animated: true }, eds));

        // Now, check the types of nodes being connected
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        // If a DataSourceNode is connected to a FeatureNode, pass the timeframe
        if (sourceNode?.type === 'dataSource' && targetNode?.type === 'feature') {
            updateNodeData(targetNode.id, { timeframe: sourceNode.data.timeframe });
        }

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

    
    const saveBacktestTemplate = useCallback(async (name: string, description: string, code: string): Promise<string> => {
        const newKey = name.toLowerCase().replace(/\s+/g, '_');
        const newTemplate: BacktestTemplate = { name, description, code, isDeletable: true };

        if (backtestTemplates[newKey]) {
            if (!window.confirm(`A backtest template named "${name}" already exists. Overwrite?`)) {
                throw new Error("Save operation cancelled by user.");
            }
        }

        await fetch(`${API_URL}/api/ml/backtest-templates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: newKey, ...newTemplate }),
        });

        setBacktestTemplates(prev => ({ ...prev, [newKey]: newTemplate }));
        return newKey;
    }, [backtestTemplates]);


    const deleteBacktestTemplate = useCallback(async (templateKey: string) => {
        if (!window.confirm(`Are you sure you want to delete the template "${backtestTemplates[templateKey]?.name}"?`)) {
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/ml/backtest-templates/${templateKey}`, {
                method: 'DELETE',
            });
            if (!response.ok) throw new Error('Failed to delete template from server.');

            setBacktestTemplates(prev => {
                const newTemplates = { ...prev };
                delete newTemplates[templateKey];
                return newTemplates;
            });

        } catch (error) {
            console.error("Error deleting backtest template:", error);
            alert("Could not delete the template.");
        }
    }, [backtestTemplates]);

    const addNode = useCallback((
        nodeType: string, 
        position: { x: number, y: number }, 
        connectingNode?: { id: string, handleType: 'source' | 'target' },
        template?: { label: string; code: string; subType: 'feature_engineering' | 'labeling' }
    ) => {
        let newNodeData: any;

        // If a template is provided for a customCode node, use it directly
        if (nodeType === 'customCode' && template) {
            newNodeData = {
                label: template.label,
                code: template.code,
                subType: template.subType // Store the subtype!
            };

        } else if (nodeType === 'customLabeling' && template) {
            newNodeData = {
                label: template.label,
                code: template.code,
                subType: template.subType // Store the subtype!
            };

        } else {
        switch (nodeType) {
            case 'feature':
            newNodeData = { label: 'Feature', name: 'SMA', params: {'length': 20} };

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
            case 'merge': 
                newNodeData = {
                    label: 'Merge',
                    mergeMethod: 'left', // Default merge strategy
                };
                break;
            case 'notes':
                newNodeData = {
                    label: 'Note Taking',
                    noteContent: 'Type your notes here...'
                };
                break;
            case 'customCode': { 
                const defaultTemplate = INITIAL_FEATURE_ENGINEERING_CODE.guide;
                newNodeData = {
                    label: defaultTemplate.name,
                    code: defaultTemplate.code,
                    subType: 'feature_engineering',
                    selectedTemplateKey: 'guide'
                };
                break;
            }
            case 'customLabeling': // This is the fallback for a blank custom node
                newNodeData = {
                    label: 'Custom Labeling',
                    code: INITIAL_FEATURE_ENGINEERING_CODE.guide.code,
                    subType: 'labeling'
                };
                break;
            case 'dataScaling': // Add the new node type case
                newNodeData = {
                    label: 'Data Scaling',
                    scaler: 'none',
                    removeCorrelated: false,
                    correlationThreshold: 0.9,
                    usePCA: false,
                    pcaComponents: 5,
                };
                break;
            case 'classImbalance':
                newNodeData = {
                    label: 'Class Imbalance',
                    method: 'SMOTE',
                };
                break;
            case 'dataValidation': // Add the new node type case
                newNodeData = {
                    label: 'Data Validation',
                    validationMethod: 'train_test_split',
                    trainSplit: 70,
                    walkForwardTrainWindow: 365,
                    walkForwardTestWindow: 30,
                };
                break;
            case 'featuresCorrelation':
                newNodeData = {
                    label: 'Feature Correlation',
                    method: 'pearson',
                    displayMode: 'matrix'
                };
                break;
            case 'charting':
                newNodeData = {
                    label: 'Charting',
                    chartType: 'scatter',
                    xAxis: null,
                    yAxis: null,
                    groupBy: null,
                };
                break;
            case 'modelTrainer': {
                const defaultModelName = 'RandomForestClassifier';
                const defaultConfig = HYPERPARAMETER_CONFIG[defaultModelName];
                const defaultHyperparameters = defaultConfig.reduce((acc, param) => {
                    acc[param.name] = param.defaultValue;
                    return acc;
                }, {} as { [key: string]: any });
                
                newNodeData = {
                    label: 'Model Trainer',
                    modelName: defaultModelName,
                    hyperparameters: defaultHyperparameters,
                    predictionThreshold: 0.5 // default threshold
                };
                break;
            }
            case 'hyperparameterTuning': {
                const defaultModelName = 'RandomForestClassifier';
                const defaultConfig = TUNING_GRID_CONFIG[defaultModelName];
                const defaultParamGrid = defaultConfig.reduce((acc, param) => {
                    acc[param.name] = param.defaultValue;
                    return acc;
                }, {} as { [key: string]: string });

                newNodeData = {
                    label: 'Hyper Parameter Tuning',
                    modelName: defaultModelName,
                    searchStrategy: 'GridSearchCV',
                    cvFolds: 5,
                    scoringMetricBase: 'accuracy', // Simple default
                    scoringMetricAvg: '',        // Empty by default
                    scoringMetricClass: '',        // Empty by default
                    paramGrid: defaultParamGrid,
                };
                break;
            }
            case 'modelPredictor':
                newNodeData = {
                    label: 'Model Predictor',
                    trainerNodeId: null, // It starts unlinked
                };
                break;
            case 'backtester': {
                newNodeData = {
                    label: 'MA Crossover Guide',
                    selectedTemplateKey: 'ma_crossover',
                    ...defaultBacktesterData
                };
                break;
            }
            }
        }
        
        const newNode: Node = {
            id: uuidv4(),
            type: nodeType,
            position,
            data: newNodeData,
        };

        if (connectingNode) {
            // This is the "create and connect" scenario.
            const newEdge: Edge = {
                id: uuidv4(),
                source: connectingNode.handleType === 'source' ? connectingNode.id : newNode.id,
                target: connectingNode.handleType === 'source' ? newNode.id : connectingNode.id,
                animated: true,
            };
            
            const sourceNode = nodes.find(n => n.id === newEdge.source);
            
            // --- Replicate ALL relevant onConnect logic here, before setting state ---
            
            // 1. Handle timeframe inheritance for Feature nodes
            if (sourceNode?.type === 'dataSource' && newNode.type === 'feature') {
                newNode.data.timeframe = sourceNode.data.timeframe;
            }

            // 2. Handle auto-selection for ProcessIndicator nodes
            if (sourceNode?.type === 'feature' && newNode.type === 'processIndicators') {
                newNode.data.selectedIndicators = {
                    ...(newNode.data.selectedIndicators || {}),
                    [sourceNode.id]: true, // Auto-check the box
                };
            }

            // Set both nodes and edges state AT ONCE.
            setNodes((nds) => [...nds, newNode]);
            setEdges((eds) => addEdge(newEdge, eds));
            
        } else {
            // This is the simple "drag from sidebar" scenario. Just add the node.
            setNodes((nds) => [...nds, newNode]);
        }
    }, [nodes, setNodes, setEdges, feTemplates]); // Add feTemplates to dependencies if needed


    const deleteElement = useCallback((id: string, type: 'node' | 'edge') => {
        if (type === 'node') {
            setEdges((eds) => eds.filter((edge) => edge.source !== id && edge.target !== id));
            setNodes((nds) => nds.filter((node) => node.id !== id));
        } else if (type === 'edge') {
            // Manually trigger the removal logic when deleting an edge directly
            onEdgesChange([{ id, type: 'remove' }]);
        }
    }, [setNodes, setEdges, onEdgesChange]); // IMPORTANT: Add onEdgesChange here

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

    // 1. Memoize selectNode
    const selectNode = useCallback((nodeId: string | null) => {
        setSelectedNodeId(nodeId);
    }, []); // Empty dependency array is correct because setSelectedNodeId is stable

    // 2. Memoize executePipelineUpToNode
    const executePipelineUpToNode = useCallback(async (targetNodeId: string) => {
        if (!workflowId) return alert("Workflow not initialized.");
        if (isProcessing) return; // Prevent concurrent runs

        setIsProcessing(true);
        setProcessingNodeId(targetNodeId);
        
        try {
            const payload = {
                // Pass the current state of nodes and edges
                nodes: nodes.map(n => ({ id: n.id, type: n.type, data: n.data })),
                edges,
                targetNodeId,
            };

            const response = await fetch(`${API_URL}/api/ml/workflow/${workflowId}/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                throw new Error(errorData.detail);
            }

            const result = await response.json();

            setPipelineNodeCache(prevCache => ({
                ...prevCache,
                ...result // Directly merge the entire results object into the cache
            }));
            selectNode(targetNodeId);

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
            alert(`Pipeline Error: ${errorMessage}`);
            setPipelineNodeCache(prevCache => ({
                ...prevCache,
                [targetNodeId]: { data: [], info: { error: errorMessage } }
            }));
            selectNode(targetNodeId);
        } finally {
            setIsProcessing(false);
            setProcessingNodeId(null);
        }
    }, [
        workflowId, 
        isProcessing, 
        nodes, // Add nodes and edges here
        edges, 
        selectNode, // This is now a stable dependency
        // State setters are stable and don't need to be in the array, but it's good practice
        setPipelineNodeCache, 
        setIsProcessing, 
        setProcessingNodeId
    ]);

    const dataForDisplay = useMemo(() => {
        const emptyDisplayData = { data: [], info: null };

        // If a node is explicitly selected, prioritize its data.
        if (selectedNodeId) {
            return pipelineNodeCache[selectedNodeId] || emptyDisplayData;
        }

        // If no node is selected, find the "last" node(s) in the graph.
        if (nodes.length > 0) {
            // Find all node IDs that are used as a source for an edge.
            const sourceNodeIds = new Set(edges.map(edge => edge.source));
            
            // A terminal node is one that is NOT a source for any edge.
            const terminalNodes = nodes.filter(node => !sourceNodeIds.has(node.id));

            let lastNodeId: string | null = null;

            if (terminalNodes.length > 0) {
                // If we have one or more terminal nodes, pick the first one.
                lastNodeId = terminalNodes[0].id;
            } else {
                // Fallback for single-node graphs or graphs with cycles.
                // Just use the first node in the list.
                lastNodeId = nodes[0].id;
            }

            // Return the cached data for the determined "last node".
            if (lastNodeId) {
                return pipelineNodeCache[lastNodeId] || emptyDisplayData;
            }
        }

        // If there are no nodes at all, return the empty state.
        return emptyDisplayData;

    }, [selectedNodeId, nodes, edges, pipelineNodeCache]); // This memo re-runs only when these dependencies change

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

        feTemplates,
        labelingTemplates,
        saveFeTemplate,
        saveLabelingTemplate,
        deleteFeTemplate,
        deleteLabelingTemplate,

        symbolList,
        isFetchingSymbols,
        indicatorSchema,
        isLoadingSchema,
        fetchError,
        workflowId,
        isProcessing,
        processingNodeId,
        selectedNodeId,
        pipelineNodeCache,
        selectNode,
        executePipelineUpToNode,
        dataForDisplay,
        newWorkflow,
        currentWorkflowName,
        savedWorkflows,
        saveWorkflow,
        loadWorkflow,
        clearBackendCache,
        editingNodeId,
        setEditingNodeId,
        backtestTemplates,
        saveBacktestTemplate,
        deleteBacktestTemplate,
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