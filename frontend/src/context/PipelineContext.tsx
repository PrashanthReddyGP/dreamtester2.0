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
import { useTerminal } from './TerminalContext';
import type { IndicatorSchema } from '../components/pipeline/types';
import { TUNING_GRID_CONFIG } from '../components/pipeline/mlModels';
import type { StrategyResult } from '../../src/services/api';
import { useAnalysis } from './AnalysisContext';

// 1. DEFINE THE SHAPE OF OUR UNDOABLE STATE
interface FlowState {
    nodes: Node[];
    edges: Edge[];
}

// 2. CREATE THE CUSTOM HOOK FOR MANAGING HISTORY
const useUndoableState = <T,>(initialState: T) => {
    const [state, setState] = useState<{
        past: T[];
        present: T;
        future: T[];
    }>({
        past: [],
        present: initialState,
        future: [],
    });

    const canUndo = state.past.length > 0;
    const canRedo = state.future.length > 0;

    const undo = useCallback(() => {
        setState((currentState) => {
            const { past, present, future } = currentState;
            if (past.length === 0) return currentState;

            const previous = past[past.length - 1];
            const newPast = past.slice(0, past.length - 1);

            return {
                past: newPast,
                present: previous,
                future: [present, ...future],
            };
        });
    }, []);

    const redo = useCallback(() => {
        setState((currentState) => {
            const { past, present, future } = currentState;
            if (future.length === 0) return currentState;

            const next = future[0];
            const newFuture = future.slice(1);

            return {
                past: [...past, present],
                present: next,
                future: newFuture,
            };
        });
    }, []);

    const set = useCallback((newState: T) => {
        setState((currentState) => {
            const { present } = currentState;

            // If the new state is the same as the present, do nothing
            if (newState === present) {
                return currentState;
            }
            
            return {
                past: [...currentState.past, present],
                present: newState,
                future: [], // A new action clears the redo stack
            };
        });
    }, []);
    
    // 1. ADD a new function to update only the "present" state for transient changes like dragging.
    // This will NOT create a new history entry.
    const setPresent = useCallback((newState: T) => {
        setState(currentState => ({
            ...currentState,
            present: newState,
        }));
    }, []);

    // A function to reset state without adding to history (for loading workflows)
    const reset = useCallback((newState: T) => {
        setState({
            past: [],
            present: newState,
            future: [],
        });
    }, []);

    return {
        state: state.present,
        setState: set,
        setPresentState: setPresent,
        resetState: reset,
        undo,
        redo,
        canUndo,
        canRedo,
    };
};

// --- Constants and Initial State (Copied from MachineLearning.tsx) ---
const API_URL = 'http://127.0.0.1:8000';




// 1. Redefine the BacktesterNodeData structure (remove indicators)
export interface BacktesterNodeData {
    label: string;
    selectedTemplateKey?: string;
    config: {
        initialCapital: number;
        riskPercent: number;
        rr: number;
        commission: number;
        exitType: 'tp_sl' | 'single_condition';
        tradeDirection: 'long' | 'short' | 'hedge';
    };
    codeBlocks: {
        entryLogic: string;
        stopLossLogic: string;
        positionSizingLogic: string;
        customExitLogic: string;
    };
}

// The template still stores a stringified version of the data
export interface BacktestTemplate {
    name: string;
    description: string;
    code: string; // This will be JSON.stringify(BacktesterNodeData)
    isDeletable: boolean;
}

// 2. Update the initial default data
const defaultBacktesterData: Omit<BacktesterNodeData, 'label' | 'selectedTemplateKey'> = {
    config: {
        initialCapital: 1000,
        riskPercent: 1,
        rr: 1,
        commission: 0.1,
        exitType: 'tp_sl',
        tradeDirection: 'short',
    },
    codeBlocks: {
        entryLogic: `# Must return a boolean.
# Available variables: i, open, high, low, close, and all indicator columns (e.g., sma_50).
short_sma[i-1] > long_sma[i-1] and short_sma[i] < long_sma[i] and open_trades[i] < 1`,
        stopLossLogic: `# Must return a price level (float).
# Available variables: i, open, high, low, close, and all indicator columns.
stop_loss[i]`,
        positionSizingLogic: `# Must return a position size (float).
# Available variables: i, capital, cash, entry_price, initial_sl, and indicators.
cash / (initial_sl - entry_price)`,
        customExitLogic: `# Must return a boolean.
# Available variables: i, j, entry_price, and all indicator columns.
close[j] > hc[i-1]`
    }
};

const INITIAL_BACKTEST_TEMPLATES: Record<string, BacktestTemplate> = {
    'short_sma_crossover': {
        name: 'Short SMA Crossover',
        description: 'A strategy template for shorting SMA crossovers.',
        code: JSON.stringify({
            label: 'Short SMA Crossover',
            ...defaultBacktesterData,
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
    onNodesChange: (changes: NodeChange[]) => void;
    onEdgesChange: (changes: EdgeChange[]) => void;
    onConnect: (connection: Connection) => void;
    reactFlowWrapperRef: React.RefObject<HTMLDivElement>;
    addNode: (nodeType: string, position: { x: number; y: number }, connectingNode?: { id: string; handleType: 'source' | 'target' }, template?: any) => Node;
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
    pipelineNodeCache: Record<string, PipelineNodeData>; // Cache for node outputs
    executePipelineUpToNode: (nodeId: string) => Promise<void>;
    dataForDisplay: { data: any[], info: any | null};
    
    newWorkflow: () => Promise<void>;
    currentWorkflowName: string | null;
    savedWorkflows: string[];
    saveWorkflow: (name: string) => Promise<void>;
    loadWorkflow: (name: string) => Promise<void>;

    clearBackendCache: () => Promise<void>;
    viewBacktestAnalysis: (nodeId: string) => void;
    
    setNavigationTarget: React.Dispatch<React.SetStateAction<string | null>>;
    navigationTarget: string | null;

    editingNodeId: string | null;
    setEditingNodeId: React.Dispatch<React.SetStateAction<string | null>>;

    undo: () => void;
    redo: () => void;
    canUndo: boolean;
    canRedo: boolean;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

export const PipelineContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const {
        state: flowState,
        setState: setFlowState,
        setPresentState: setPresentFlowState,
        resetState: resetFlowState,
        undo,
        redo,
        canUndo,
        canRedo,
    } = useUndoableState<FlowState>({
        nodes: initialNodes,
        edges: initialEdges,
    });

    const { nodes, edges } = flowState; // Destructure for convenience

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
    const [pipelineNodeCache, setPipelineNodeCache] = useState<Record<string, PipelineNodeData>>({});

    const [feTemplates, setFeTemplates] = useState<Record<string, FETemplate>>(INITIAL_FEATURE_ENGINEERING_CODE);
    const [labelingTemplates, setLabelingTemplates] = useState<Record<string, LabelingTemplate>>(INITIAL_LABELING_TEMPLATES);
    const [backtestTemplates, setBacktestTemplates] = useState<Record<string, BacktestTemplate>>(INITIAL_BACKTEST_TEMPLATES);

    const [savedWorkflows, setSavedWorkflows] = useState<string[]>([]);

    const [currentWorkflowName, setCurrentWorkflowName] = useState<string | null>(null);

    // Add the new state for tracking the editing node
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
    
    const { connectToBatch, toggleTerminal } = useTerminal();

    const [navigationTarget, setNavigationTarget] = useState<string | null>(null);

    // Instantiate hooks for analysis and navigation
    const { setSingleResult } = useAnalysis();

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
                resetFlowState({ nodes: workflow.nodes, edges: workflow.edges });
                // Reset runtime state
                setPipelineNodeCache({});
            } else {
                throw new Error("Invalid workflow data received from server");
            }
        } catch (error) {
            console.error("Error loading workflow:", error);
            alert(`Error: Could not load workflow. ${error instanceof Error ? error.message : ''}`);
        }
    }, [resetFlowState]);

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
        let currentNodes = nodes;
        let currentEdges = edges;

        // Step 1: Apply the primary update to the target node
        const newNds = currentNodes.map((node) =>
            node.id === nodeId
                ? { ...node, data: { ...node.data, ...data } }
                : node
        );

        // Step 2: Check for timeframe propagation
        const sourceNode = newNds.find(n => n.id === nodeId);
        if (sourceNode?.type === 'dataSource' && 'timeframe' in data) {
            const newTimeframe = data.timeframe;
            const childEdges = currentEdges.filter(e => e.source === nodeId);
            const childIds = new Set(childEdges.map(e => e.target));
            
            const finalNodes = newNds.map(node => {
                if (childIds.has(node.id) && node.type === 'feature') {
                    return {
                        ...node,
                        data: { ...node.data, timeframe: newTimeframe }
                    };
                }
                return node;
            });
            setFlowState({ nodes: finalNodes, edges: currentEdges });
        } else {
            // If no propagation, just update the nodes
            setFlowState({ nodes: newNds, edges: currentEdges });
        }
    }, [nodes, edges, setFlowState]); 

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
            
            resetFlowState({ nodes: initialNodes, edges: initialEdges });
            setPipelineNodeCache({});
            setCurrentWorkflowName(null);
        } catch (error) {
            console.error("Failed to clear backend cache:", error);
            alert("Error: Could not clear backend cache.");
        }
    }, [resetFlowState]);

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
    const onNodesChange = useCallback((changes: NodeChange[]) => {
        const nextNodes = applyNodeChanges(changes, nodes);

        // Check if the change is a drag event. A drag event is a series of position changes.
        // The last position change in a drag has `dragging: false`. All others have `dragging: true`.
        const isDrag = changes.every(c => c.type === 'position' && c.dragging === true);

        if (isDrag) {
            // If we are dragging, we only update the 'present' state for smooth visuals
            // without creating a new history entry for every pixel moved.
            setPresentFlowState({ nodes: nextNodes, edges });
        } else {
            // For any other change (selection, node drop, dimension change), we commit it to history.
            setFlowState({ nodes: nextNodes, edges });
        }
    }, [nodes, edges, setFlowState, setPresentFlowState]);

    const onEdgesChange = useCallback((changes: EdgeChange[]) => {
        // Find which edges are being removed *before* applying the state change
        const removedEdges = changes
            .filter((c): c is { id: string; type: 'remove' } => c.type === 'remove')
            .map(c => edges.find(e => e.id === c.id))
            .filter((e): e is Edge => e !== undefined);

        const nextEdges = applyEdgeChanges(changes, edges);
        let nextNodes = nodes;

        // Now, process the logic for the removed edges
        removedEdges.forEach(removedEdge => {
            const sourceNode = nextNodes.find(n => n.id === removedEdge.source);
            const targetNode = nextNodes.find(n => n.id === removedEdge.target);

            if (sourceNode?.type === 'feature' && targetNode?.type === 'processIndicators') {
                const currentSelections = targetNode.data.selectedIndicators || {};
                const { [sourceNode.id]: _, ...remainingSelections } = currentSelections;
                
                // We map over nextNodes to create an entirely new array
                nextNodes = nextNodes.map(n => 
                    n.id === targetNode.id 
                    ? { ...n, data: { ...n.data, selectedIndicators: remainingSelections } }
                    : n
                );
            }
        });
        
        setFlowState({ nodes: nextNodes, edges: nextEdges });

    }, [edges, nodes, setFlowState]);

    // Validation Function
    const isValidConnection = (connection: Connection): boolean => {
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) return false;

        // // RULE 1: ClassImbalanceNode can only connect to the 'train' output of a DataValidationNode.
        // if (targetNode.type === 'classImbalance') {
        //     if (sourceNode.type !== 'dataValidation' || connection.sourceHandle !== 'train') {
        //         alert("Invalid Connection:\nThe Class Imbalance node can only be connected to the 'Train' output of a Data Validation node.");
        //         return false;
        //     }
        // }

        // // RULE 2: ModelTrainerNode's 'test' handle MUST connect to a DataValidationNode's 'test' output.
        // if (targetNode.type === 'modelTrainer' && connection.targetHandle === 'test') {
        //     if (sourceNode.type !== 'dataValidation' || connection.sourceHandle !== 'test') {
        //         alert("Invalid Connection:\nThe Model Trainer's 'Test Data' input must be connected to the 'Test' output of a Data Validation node.");
        //         return false;
        //     }
        // }

        // Add more rules here in the future...

        return true; // All checks passed
    };

    const onConnect = useCallback((connection: Connection) => {
        // Call the validation function first
        if (!isValidConnection(connection)) {
            return;
        }

        const nextEdges = addEdge({ ...connection, animated: true }, edges);
        let nextNodes = nodes;

        const sourceNode = nextNodes.find(n => n.id === connection.source);
        const targetNode = nextNodes.find(n => n.id === connection.target);

        // Logic for timeframe propagation
        if (sourceNode?.type === 'dataSource' && targetNode?.type === 'feature') {
            nextNodes = nextNodes.map(n => 
                n.id === targetNode.id
                ? { ...n, data: { ...n.data, timeframe: sourceNode.data.timeframe } }
                : n
            );
        }

        // Logic for process indicator selection
        if (sourceNode?.type === 'feature' && targetNode?.type === 'processIndicators') {
            const updatedSelection = {
                ...(targetNode.data.selectedIndicators || {}),
                [sourceNode.id]: true,
            };
            nextNodes = nextNodes.map(n =>
                n.id === targetNode.id
                ? { ...n, data: { ...n.data, selectedIndicators: updatedSelection } }
                : n
            );
        }

        setFlowState({ nodes: nextNodes, edges: nextEdges });

    }, [nodes, edges, setFlowState]);
    
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
        template?: any // Changed type to `any` to allow pasted node data
    ): Node => { // Return type is still Node
        ///// FIX ///// - REFACTOR TO USE setFlowState
        let newNodeData: any;

        // If a template/initial data is provided, use it. This will handle both menu templates and pasted data.
        if (template) {
            newNodeData = template;
        
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
            case 'loop':
                newNodeData = {
                    label: 'Loop',
                    variableName: 'sma_length',
                    loopType: 'numeric_range',
                    numericStart: 10,
                    numericEnd: 50,
                    numericStep: 10,
                    valueList: '10,20,50' // Provide a default for the other type too
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
                    code: INITIAL_LABELING_TEMPLATES.guide.code, // Fixed to use labeling template
                    subType: 'labeling'
                };
                break;
            case 'dataProfiler':
                newNodeData = {
                    label: 'Data Profiler',
                    selectedFeature: null, // Initially, no feature is selected
                };
                break;
            case 'dataScaling':
                newNodeData = {
                    label: 'Data Scaling',
                    scaler: 'none',
                    removeCorrelated: false,
                    correlationThreshold: 0.9,
                    usePCA: false,
                    pcaComponents: 5,
                };
                break;
            case 'advancedDataScaling':
                newNodeData = {
                    label: 'Advanced Data Scaling',
                    standardFeatures: [],
                    minmaxFeatures: [],
                    robustFeatures: [],
                    removeCorrelated: false,
                    isConfigured: false,
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
            case 'neuralNetworkTrainer': {
                newNodeData = {
                    label: 'Neural Network',
                    predictionThreshold: 0.5,
                    architecture: {
                        layers: [
                            { id: 'layer1', type: 'Dense', units: 64, activation: 'relu' },
                            { id: 'layer2', type: 'Dropout', rate: 0.5 },
                            { id: 'layer3', type: 'Dense', units: 32, activation: 'relu' },
                        ]
                        },
                        training: {
                        optimizer: 'adam',
                        loss: 'binary_crossentropy',
                        epochs: 50,
                        batchSize: 32,
                        earlyStoppingPatience: 10
                        }
                    };
                break;
            }
            case 'baggingTrainer': {
                const defaultBaseModel = 'DecisionTreeClassifier';
                const baseConfig = HYPERPARAMETER_CONFIG[defaultBaseModel];
                const baggingConfig = HYPERPARAMETER_CONFIG['BaggingClassifier'];

                newNodeData = {
                    label: 'Bagging Trainer',
                    predictionThreshold: 0.5,
                    baseModelName: defaultBaseModel,
                    baseModelHyperparameters: baseConfig.reduce((acc, param) => {
                        acc[param.name] = param.defaultValue;
                        return acc;
                    }, {} as { [key: string]: any }),
                    baggingHyperparameters: baggingConfig.reduce((acc, param) => {
                        acc[param.name] = param.defaultValue;
                        return acc;
                    }, {} as { [key: string]: any }),
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
                    label: 'Short SMA Crossover',
                    selectedTemplateKey: 'short_sma_crossover',
                    ...defaultBacktesterData
                };
                break;
            }
            }
        }
        
        const newNode: Node = { id: uuidv4(), type: nodeType, position, data: newNodeData, };

        let nextNodes = [...nodes, newNode];
        let nextEdges = edges;

        if (connectingNode) {
            const newEdge: Edge = { id: uuidv4(), source: connectingNode.handleType === 'source' ? connectingNode.id : newNode.id, target: connectingNode.handleType === 'source' ? newNode.id : connectingNode.id, animated: true, };
            nextEdges = addEdge(newEdge, edges);
            const sourceNode = nodes.find(n => n.id === newEdge.source);
            
            if (sourceNode?.type === 'dataSource' && newNode.type === 'feature') {
                newNode.data.timeframe = sourceNode.data.timeframe; // Modify newNode directly before adding
            }
            if (sourceNode?.type === 'feature' && newNode.type === 'processIndicators') {
                newNode.data.selectedIndicators = { ...(newNode.data.selectedIndicators || {}), [sourceNode.id]: true, };
            }
        }
        
        setFlowState({ nodes: nextNodes, edges: nextEdges });
        return newNode;
    }, [nodes, edges, setFlowState]);

    const deleteElement = useCallback((id: string, type: 'node' | 'edge') => {
        if (type === 'node') {
            const nextEdges = edges.filter((edge) => edge.source !== id && edge.target !== id);
            const nextNodes = nodes.filter((node) => node.id !== id);
            setFlowState({ nodes: nextNodes, edges: nextEdges });
        } else if (type === 'edge') {
            // Re-use onEdgesChange logic which is now history-aware
            onEdgesChange([{ id, type: 'remove' }]);
        }
    }, [nodes, edges, setFlowState, onEdgesChange]);

    // Function to handle viewing backtest results
    const viewBacktestAnalysis = useCallback((nodeId: string) => {
        const nodeCache = pipelineNodeCache[nodeId];
        
        const backtestResult = nodeCache?.info?.backtest_result as StrategyResult;

        if (backtestResult) {
            console.log("Found pipeline backtest result, setting single result for analysis:", backtestResult);
            
            // --- Use the new, single, synchronous function ---
            setSingleResult(backtestResult, { test_type: 'pipeline_backtest' });
            
            // Set the navigation target. The useEffect in PipelineEditor will handle the rest.
            setNavigationTarget('/analysis');
        
        } else {
            alert("No backtest results found for this node. Please run the node first.");
            console.warn(`Analysis button clicked, but no result in cache for node ${nodeId}. Cache content:`, nodeCache);
        }
    }, [pipelineNodeCache, setSingleResult]); // Update dependencies

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

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
            alert(`Pipeline Error: ${errorMessage}`);
            setPipelineNodeCache(prevCache => ({
                ...prevCache,
                [targetNodeId]: { data: [], info: { error: errorMessage } }
            }));
        } finally {
            setIsProcessing(false);
            setProcessingNodeId(null);
        }
    }, [
        workflowId, 
        isProcessing, 
        nodes, // Add nodes and edges here
        edges, 
        // State setters are stable and don't need to be in the array, but it's good practice
        setPipelineNodeCache, 
        setIsProcessing, 
        setProcessingNodeId
    ]);

    const dataForDisplay = useMemo(() => {
        const emptyDisplayData = { data: [], info: null };
        
        // Find all selected nodes.
        const selectedNodes = nodes.filter(node => node.selected);

        // If exactly one node is selected, show its data.
        if (selectedNodes.length === 1) {
            const selectedId = selectedNodes[0].id;
            return pipelineNodeCache[selectedId] || emptyDisplayData;
        }

        // If more than one node is selected, we could show a summary or nothing.
        // For now, let's just show an info message in the panel.
        if (selectedNodes.length > 1) {
            return {
                data: [],
                info: { message: `${selectedNodes.length} nodes selected.` }
            };
        }

        // If no nodes are selected, return the default empty state.
        return emptyDisplayData;

    }, [nodes, pipelineNodeCache]);

    const value = {
        nodes,
        edges,
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
        pipelineNodeCache,
        executePipelineUpToNode,
        dataForDisplay,
        newWorkflow,
        currentWorkflowName,
        savedWorkflows,
        saveWorkflow,
        loadWorkflow,
        clearBackendCache,
        viewBacktestAnalysis,
        navigationTarget,    
        setNavigationTarget,  
        editingNodeId,
        setEditingNodeId,
        backtestTemplates,
        saveBacktestTemplate,
        deleteBacktestTemplate,
        undo,
        redo,
        canUndo,
        canRedo,
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