// PipelineEditor.tsx
import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import { Box, Menu, MenuItem, IconButton, ListItemText } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete'; // Import the delete icon
import ReactFlow, {
    Controls as ReactFlowControls, // Import with an alias
    Background,
    ReactFlowProvider,
    useReactFlow,
} from 'reactflow';
import type { Node, Edge } from 'reactflow'; // Add Node, Edge here
import { useNavigate } from 'react-router-dom'; // Add useNavigate
import 'reactflow/dist/style.css';
import { styled } from '@mui/material/styles'; // Import styled
import NestedMenuItem from '../components/common/NestedMenuItem'; // Adjust path if necessary

import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';

import { DataSourceNode } from '../components/pipeline/nodes/DataSourceNode';
import { FeatureNode } from '../components/pipeline/nodes/FeatureNode';
import { usePipeline } from '../context/PipelineContext';
import { SaveTemplateDialog } from '../components/common/SaveTemplateDialog'; // 1. Import the new dialog

import {
    Panel,
    PanelGroup,
    PanelResizeHandle,
} from "react-resizable-panels";
import { SidePanel, CollapsedPanelHandle } from '../components/pipeline/SidePanel';
import { ProcessIndicatorsNode } from '../components/pipeline/nodes/ProcessIndicatorsNode';
import { EditorNode } from '../components/pipeline/nodes/EditorNode';
import { NotesNode } from '../components/pipeline/nodes/NotesNode';
import { LabelingNode } from '../components/pipeline/nodes/LabelingNode';
import { ModelTrainerNode } from '../components/pipeline/nodes/ModelTrainerNode';
import { ModelPredictorNode } from '../components/pipeline/nodes/ModelPredictorNode';
import type { FETemplate, LabelingTemplate  } from '../context/PipelineContext';
import { DataScalingNode } from '../components/pipeline/nodes/DataScalingNode';
import { DataValidationNode } from '../components/pipeline/nodes/DataValidationNode';
import { ChartingNode } from '../components/pipeline/nodes/ChartingNode';
import { PipelineControls } from '../components/pipeline/PipelineControls';
import { MergeNode } from '../components/pipeline/nodes/MergeNode';
import { HyperparameterTuningNode } from '../components/pipeline/nodes/HyperparameterTuningNode';
import { ClassImbalanceNode } from '../components/pipeline/nodes/ClassImbalanceNode';
import { BacktesterNode } from '../components/pipeline/nodes/BacktesterNode';
import { BaggingTrainerNode } from '../components/pipeline/nodes/BaggingTrainerNode';
import EditIcon from '@mui/icons-material/Edit';
import { FeaturesCorrelationNode } from '../components/pipeline/nodes/FeaturesCorrelationNode';
import { LoopNode } from '../components/pipeline/nodes/LoopNode';
import { NeuralNetworkTrainerNode } from '../components/pipeline/nodes/NeuralNetworkTrainerNode';
import { DataProfilerNode } from '../components/pipeline/nodes/DataProfilerNode';
import { AdvancedDataScalingNode } from '../components/pipeline/nodes/AdvancedDataScalingNode';

const StyledControls = styled(ReactFlowControls)(({ theme }) => ({
    backgroundColor: 'theme.palette.background.paper',
    borderRadius: '12px',
    boxShadow: 'none',
    
    // Target the individual buttons
    '& .react-flow__controls-button': {
        backgroundColor: 'transparent',
        border: 'none',
        padding: theme.spacing(1.25), // Controls the size/internal space of the buttons. Adjust as needed.
        margin: 0,                    // Removes the space *between* the buttons.
        borderRadius: 0,              // Ensures buttons are perfect rectangles inside the container.

        '&:hover': {
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        },

        // We must target the SVG element inside the button directly
        // and use the 'fill' property instead of 'color'.
        '& svg': {
            fill: theme.palette.common.white,
        },
    },
}));


const PipelineEditorContent: React.FC = () => {
    const {
        nodes,
        edges,
        onNodesChange,
        onEdgesChange,
        onConnect,
        reactFlowWrapperRef,
        addNode,
        deleteElement,
        symbolList,
        isFetchingSymbols,
        indicatorSchema,
        isLoadingSchema,
        pipelineNodeCache,
        isProcessing,
        processingNodeId,
        executePipelineUpToNode,
        dataForDisplay,
        feTemplates,
        labelingTemplates,
        saveFeTemplate,
        saveLabelingTemplate,
        deleteFeTemplate,
        deleteLabelingTemplate,
        navigationTarget,      
        setNavigationTarget,  
        editingNodeId,
        setEditingNodeId,
        undo,
        redo,
        canUndo,
        canRedo,
    } = usePipeline();

    const processedEdges = useMemo(() => {
        // Create a quick lookup map for node positions
        const nodesMap = new Map(nodes.map(node => [node.id, node]));

        return edges.map(edge => {
            const sourceNode = nodesMap.get(edge.source);
            const targetNode = nodesMap.get(edge.target);

            // A simple heuristic for detecting a "feedback" loop:
            // If the source node is positioned to the right of the target node.
            if (sourceNode && targetNode && sourceNode.position.x > targetNode.position.x) {
                return {
                    ...edge,
                    type: 'smoothstep', // Assign the smoothstep type for feedback loops
                    pathOptions: {
                        borderRadius: 16,
                    }
                };
            }

            // For all other "forward" connections, use the default bezier curve.
            // By not setting a type, it will use the default.
            return edge;
        });
    }, [nodes, edges]);

    const { project, getNodes, getEdges } = useReactFlow();
    const navigate = useNavigate();

    // State for our clipboard
    const [copiedNode, setCopiedNode] = useState<Node | null>(null);

    const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);

    // Keyboard shortcut handler
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            const isModifierPressed = event.ctrlKey || event.metaKey;

            // Use event.target instead of document.activeElement for a more reliable check.
            const target = event.target as HTMLElement;

            // Use .closest() to check if the event originated from within an input,
            // a textarea, or any content-editable element. This is much more reliable
            // than checking event.target alone, especially for complex components.
            const isTyping = target.closest(
                'input, textarea, [contenteditable="true"]'
            );
            
            // The rest of your logic remains the same.
            if (isTyping) {
                // A special case for the 'Escape' key. We might want to allow it to
                // bubble up so it can be used to unfocus an input.
                if (event.key === 'Escape') {
                    // Optionally let escape bubble up
                } else {
                // Otherwise, do nothing and let the browser handle the typing.
                return;
                }
            }

            // Handle Delete/Backspace for nodes and edges
            if (event.key === 'Delete' || event.key === 'Backspace') {
                event.preventDefault();
                const selectedNodes = getNodes().filter((n) => n.selected);
                const selectedEdges = getEdges().filter((e) => e.selected);
                if (selectedNodes.length > 0 || selectedEdges.length > 0) {
                    const nodeIdsToDelete = selectedNodes.map(n => n.id);
                    const edgeIdsToDelete = selectedEdges.map(e => e.id);
                    nodeIdsToDelete.forEach(id => deleteElement(id, 'node'));
                    edgeIdsToDelete.forEach(id => deleteElement(id, 'edge'));
                }
            }

            // Handle Ctrl/Cmd shortcuts
            if (isModifierPressed) {
                switch (event.key.toLowerCase()) {
                    case 'c': {
                        // --- THE FIX IS HERE ---
                        const selection = window.getSelection();
                        const hasTextSelection = selection && selection.toString().length > 0;

                        // If the user has selected any text on the page, DO NOTHING.
                        // Let the browser perform its default copy action.
                        if (hasTextSelection) {
                            break;
                        }
                        
                        // Otherwise, if no text is selected, proceed with our custom "copy node" logic.
                        event.preventDefault(); // Now it's safe to prevent default
                        const selectedNode = getNodes().find((n) => n.selected);
                        if (selectedNode) {
                            setCopiedNode(selectedNode);
                        }
                        break;
                    }
                    case 'v': {
                        event.preventDefault();
                        if (copiedNode) {
                            const newPosition = { x: copiedNode.position.x + 30, y: copiedNode.position.y + 30 };
                            const newData = { ...copiedNode.data, label: `${copiedNode.data.label} (Copy)` };
                            addNode(copiedNode.type!, newPosition, undefined, newData);
                        }
                        break;
                    }
                    case 'z': {
                        event.preventDefault();
                        if (event.shiftKey) {
                            if (canRedo) redo();
                        } else {
                            if (canUndo) undo();
                        }
                        break;
                    }
                }
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [addNode, deleteElement, getNodes, getEdges, copiedNode, undo, redo, canUndo, canRedo]);

    const [menu, setMenu] = useState<{
        id: string | null;
        type: 'node' | 'edge' | 'pane' | 'connect-end';
        top: number;
        left: number;
        connectingNodeId?: string;
        connectingHandleType?: 'source' | 'target';
    } | null>(null);

    const connectingNodeId = useRef<string | null>(null);
    const connectingHandleType = useRef<string | null>(null);

    const [isPanelOpen, setIsPanelOpen] = useState(true);
    const [panelPosition, setPanelPosition] = useState<'left' | 'right' | 'top' | 'bottom'>('right');
    
    useEffect(() => {
        // If navigationTarget has a path...
        if (navigationTarget) {
            // ...navigate to it.
            navigate(navigationTarget);
            // And IMMEDIATELY reset the target to null to prevent re-navigation
            // if this component re-renders for other reasons.
            setNavigationTarget(null);
        }
    }, [navigationTarget, navigate, setNavigationTarget]); // Dependencies

    // Effect to aggressively prevent default middle-mouse button behavior (auto-scroll)
    // using the event capture phase to eliminate input lag.
    useEffect(() => {
        const rfWrapper = reactFlowWrapperRef.current;
        if (!rfWrapper) return;

        const handleMouseDown = (event: MouseEvent) => {
            // Check if the middle mouse button is pressed (event.button === 1)
            if (event.button === 1) {
                // ONLY prevent the default browser action (autoscroll).
                // DO NOT stop propagation, as that prevents React Flow from getting the event.
                event.preventDefault(); 
            }
        };

        // Add the event listener in the CAPTURE phase.
        rfWrapper.addEventListener('mousedown', handleMouseDown, { capture: true });

        // The cleanup function MUST also use the same capture option.
        return () => {
            rfWrapper.removeEventListener('mousedown', handleMouseDown, { capture: true });
        };
    }, []); // Empty dependency array is correct.


    const onConnectStart = useCallback((_, { nodeId, handleType }) => {
        connectingNodeId.current = nodeId;
        connectingHandleType.current = handleType as 'source' | 'target' | null;
    }, []);

    const onConnectEnd = useCallback(
        (event: MouseEvent | TouchEvent) => {
            const target = event.target as Element;
            const targetIsPane = target.classList.contains('react-flow__pane');

            if (targetIsPane && connectingNodeId.current && connectingHandleType.current) {
                let clientX, clientY;
                if ('clientX' in event) {
                    clientX = event.clientX;
                    clientY = event.clientY;
                } else {
                    clientX = event.changedTouches[0].clientX;
                    clientY = event.changedTouches[0].clientY;
                }
                setMenu({
                    id: null,
                    type: 'connect-end',
                    top: clientY,
                    left: clientX,
                    connectingNodeId: connectingNodeId.current,
                    connectingHandleType: connectingHandleType.current,
                });
            }
        },
        []
    );

    const onContextMenu = useCallback(
        (event: React.MouseEvent, element?: Node | Edge) => {
            event.preventDefault();
            const type = element ? ('source' in element ? 'edge' : 'node') : 'pane';
            const id = element ? element.id : null;

            setMenu({
                id,
                type,
                top: event.clientY,
                left: event.clientX,
            });
        },
        []
    );

    const handleClose = () => setMenu(null);
    
    const handleAddNode = (nodeType: string) => {
        if (!menu) return;
        const menuState = { ...menu };
        handleClose();

        const position = project({ x: menuState.left, y: menuState.top });
        
        const connectingNode = menuState.type === 'connect-end' && menuState.connectingNodeId && menuState.connectingHandleType
            ? { id: menuState.connectingNodeId, handleType: menuState.connectingHandleType }
            : undefined;

        addNode(nodeType, position, connectingNode);
    };

    const handleRenameNode = () => { if (!menu || !menu.id || menu.type !== 'node') return; setEditingNodeId(menu.id); handleClose(); };

    // We memoize the nodeTypes object to prevent it from being recreated on every render.
    // The dependency array is stable because the lists and schemas only load once.
    // This allows us to pass the necessary props down to each node type.
    const nodeTypes = useMemo(() => ({
        dataSource: (props: any) => (
            <DataSourceNode
                {...props}
                symbolList={symbolList}
                isFetchingSymbols={isFetchingSymbols}
            />
        ),
        feature: (props: any) => (
            <FeatureNode
                {...props}
                indicatorSchema={indicatorSchema}
                isLoadingSchema={isLoadingSchema}
            />
        ),
        // Simple nodes don't need extra props
        processIndicators: ProcessIndicatorsNode,
        merge: MergeNode,
        notes: NotesNode,
        loop: LoopNode,
        customCode: EditorNode,
        customLabeling: LabelingNode,
        dataProfiler: DataProfilerNode,
        dataScaling: DataScalingNode,
        advancedDataScaling: AdvancedDataScalingNode,
        dataValidation: DataValidationNode,
        classImbalance: ClassImbalanceNode,
        charting: ChartingNode,
        featuresCorrelation: FeaturesCorrelationNode,
        modelTrainer: ModelTrainerNode,
        neuralNetworkTrainer: NeuralNetworkTrainerNode,
        hyperparameterTuning: HyperparameterTuningNode, 
        modelPredictor: ModelPredictorNode,
        backtester: BacktesterNode,
        baggingTrainer: BaggingTrainerNode,
    }), [symbolList, isFetchingSymbols, indicatorSchema, isLoadingSchema]); // Stable dependencies


    const handleDeleteElement = () => {
        if (!menu || !menu.id || (menu.type !== 'node' && menu.type !== 'edge')) return;
        handleClose();
        deleteElement(menu.id, menu.type);
    };

    const handleAddNodeFromTemplate = (
        template: FETemplate | LabelingTemplate, 
        subType: 'feature_engineering' | 'labeling'
    ) => {
        if (!menu) return;
        const menuState = { ...menu };
        handleClose();

        const position = project({ x: menuState.left, y: menuState.top });
        const connectingNode = menuState.type === 'connect-end' && menuState.connectingNodeId && menuState.connectingHandleType
            ? { id: menuState.connectingNodeId, handleType: menuState.connectingHandleType }
            : undefined;
        
        if (subType === 'feature_engineering') {
            // Call the modified addNode with the template data
            addNode('customCode', position, connectingNode, {
                label: template.name,
                code: template.code,
                subType: subType
            }); 
        } else if (subType === 'labeling') {
            // Call the modified addNode with the template data
            addNode('customLabeling', position, connectingNode, {
                label: template.name,
                code: template.code,
                subType: subType
            });
        }
    };

    // This function now just OPENS the dialog.
    const handleSaveTemplate = () => {
        if (!menu?.id) return;
        const nodeToSave = nodes.find(n => n.id === menu.id);
        // Only allow saving templates for custom code or labeling nodes
        if (!nodeToSave || (nodeToSave.type !== 'customCode' && nodeToSave.type !== 'customLabeling')) return;
        
        setIsSaveDialogOpen(true); // Open the dialog
    };
    
    // --- 4. CREATE A NEW HANDLER FOR THE DIALOG'S "SAVE" ACTION ---
    const handleConfirmSave = (templateName: string, description: string) => {
        if (!menu?.id) return;
        const nodeToSave = nodes.find(n => n.id === menu.id);
        if (!nodeToSave) return; // Should not happen if dialog was opened
        
        const { code, subType } = nodeToSave.data;
        
        if (subType === 'feature_engineering') {
            saveFeTemplate(templateName, description, code);
        } else if (subType === 'labeling') {
            saveLabelingTemplate(templateName, description, code);
        }
        
        handleClose(); // Close the right-click menu
    };

    const handleCollapse = () => { setIsPanelOpen(false); };
    const handleExpand = () => { setIsPanelOpen(true); };
    const togglePanel = () => {
        setIsPanelOpen(prev => !prev);
    };

    const isHorizontal = panelPosition === 'left' || panelPosition === 'right';

    const panelComponent = ( 
        <Panel collapsible={true} collapsedSize={0} defaultSize={25} minSize={15} onCollapse={handleCollapse} onExpand={handleExpand} > 
            <SidePanel 
                isPanelOpen={isPanelOpen} 
                panelPosition={panelPosition} 
                setPanelPosition={setPanelPosition} 
                togglePanel={togglePanel} 
                displayData={dataForDisplay.data} 
                displayInfo={dataForDisplay.info} 
                selectedNode={nodes.find(n => n.selected)} /> 
        </Panel> 
    );

    const reactFlowComponent = (
        <Panel defaultSize={62} minSize={30}>
            <Box sx={{ width: '100%', height: '100%' }} ref={reactFlowWrapperRef}>
                <ReactFlow
                    nodes={nodes}
                    edges={processedEdges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onConnectStart={onConnectStart}
                    onConnectEnd={onConnectEnd}
                    nodeTypes={nodeTypes}
                    onPaneContextMenu={(e) => onContextMenu(e)}
                    onNodeContextMenu={(e, node) => onContextMenu(e, node)}
                    onEdgeContextMenu={(e, edge) => onContextMenu(e, edge)}
                    
                    elevateNodesOnSelect={true}
                    fitView
                    minZoom={0.2}
                    maxZoom={1}
                    panOnScroll={false}
                    selectionOnDrag={true}
                    panOnDrag={[1]}
                    connectionLineStyle={{
                        strokeWidth: 2,
                        stroke: '#eaeaeaff',
                    }}
                    defaultEdgeOptions={{
                        style: { strokeWidth: 2, stroke: '#eaeaeaff' },
                        animated: true,
                    }}
                    proOptions={{ hideAttribution: true }}
                    >
                    <StyledControls position="top-right" />
                    <Background />
                </ReactFlow>
            </Box>
        </Panel>
    );

    return (
        <Box sx={{ width: '100vw', height: 'calc(100vh - 88px)', position: 'relative' }}>
            <PipelineControls />
            {isPanelOpen ? (
                <PanelGroup direction={isHorizontal ? 'horizontal' : 'vertical'}>
                    {(panelPosition === 'left' || panelPosition === 'top') && panelComponent}
                    {isPanelOpen && (panelPosition === 'left' || panelPosition === 'top') && (
                        <PanelResizeHandle></PanelResizeHandle>
                    )}
                    {reactFlowComponent}
                    {isPanelOpen && (panelPosition === 'right' || panelPosition === 'bottom') && (
                        <PanelResizeHandle></PanelResizeHandle>
                    )}
                    {(panelPosition === 'right' || panelPosition === 'bottom') && panelComponent}
                </PanelGroup>
            ) : (
                <>
                    <Box sx={{ width: '100%', height: '100%', overflow: 'hidden' }} ref={reactFlowWrapperRef}>
                        <ReactFlow
                            nodes={nodes}
                            edges={edges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            onConnect={onConnect}
                            onConnectStart={onConnectStart}
                            onConnectEnd={onConnectEnd}
                            nodeTypes={nodeTypes}
                            onPaneContextMenu={(e) => onContextMenu(e)}
                            onNodeContextMenu={(e, node) => onContextMenu(e, node)}
                            onEdgeContextMenu={(e, edge) => onContextMenu(e, edge)}
                            
                            elevateNodesOnSelect={true}
                            fitView
                            minZoom={0.2}
                            maxZoom={1.2}
                            panOnScroll={false}
                            selectionOnDrag={true}
                            panOnDrag={[1]}
                            connectionLineStyle={{
                                strokeWidth: 2,
                                stroke: '#eaeaeaff',
                            }}
                            defaultEdgeOptions={{
                                style: { strokeWidth: 2, stroke: '#eaeaeaff' },
                                animated: true,
                            }}
                            proOptions={{ hideAttribution: true }}
                            >
                            <StyledControls position="top-right" />
                            <Background />
                        </ReactFlow>
                    </Box>
                    <CollapsedPanelHandle position={panelPosition} onToggle={togglePanel} />
                </>
            )}
            <SaveTemplateDialog
                open={isSaveDialogOpen}
                onClose={() => {
                    setIsSaveDialogOpen(false);
                    handleClose(); // Also close the context menu
                }}
                onSave={handleConfirmSave}
                title="Save Node as Template"
            />
            <Menu
                open={menu !== null}
                onClose={handleClose}
                anchorReference="anchorPosition"
                anchorPosition={
                    menu !== null ? { top: menu.top, left: menu.left } : undefined
                }
            >
                {(menu?.type === 'pane' || menu?.type === 'connect-end') && [
                    <MenuItem key="add-ds" onClick={() => handleAddNode('dataSource')}>Add Data Source</MenuItem>,
                    <MenuItem key="add-feat" onClick={() => handleAddNode('feature')}>Add Feature</MenuItem>,
                    <MenuItem key="add-proc" onClick={() => handleAddNode('processIndicators')}>Add Indicator Processor</MenuItem>,
                    <MenuItem key="add-merge" onClick={() => handleAddNode('merge')}>Add Merge Node</MenuItem>,
                    <MenuItem key="add-notes" onClick={() => handleAddNode('notes')}>Add Notes</MenuItem>,
                    <MenuItem key="add-loop" onClick={() => handleAddNode('loop')}>Add Loop Node</MenuItem>,
                    <MenuItem key="add-editor" onClick={() => handleAddNode('customCode')}>Add Editor Node</MenuItem>,
                    <NestedMenuItem
                        key="fe-templates"
                        label="Feature Engineering"
                        parentMenuOpen={!!menu}
                    >
                        {Object.entries(feTemplates).map(([key, template]) => (
                            <MenuItem 
                                key={`fe-${key}`} 
                                // We stop the event propagation on the delete button,
                                // so clicking it won't trigger this onClick for the whole item.
                                onClick={() => {
                                    handleAddNodeFromTemplate(template, 'feature_engineering');
                                    handleClose();
                                }}
                                sx={{ display: 'flex', justifyContent: 'space-between' }}
                            >
                                <ListItemText>{template.name}</ListItemText>
                                {template.isDeletable && (
                                    <IconButton
                                        size="small"
                                        onClick={(e) => {
                                            e.stopPropagation(); // Stop the click from bubbling up to the MenuItem
                                            deleteFeTemplate(key);
                                            handleClose(); // Close menu after deletion
                                        }}
                                        aria-label={`delete ${template.name}`}
                                    >
                                        <DeleteIcon fontSize="small" />
                                    </IconButton>
                                )}
                            </MenuItem>
                        ))}
                    </NestedMenuItem>,

                    <NestedMenuItem
                        key="label-templates"
                        label="Labeling"
                        parentMenuOpen={!!menu}
                    >
                        {Object.entries(labelingTemplates).map(([key, template]) => (
                            <MenuItem 
                                key={`label-${key}`} 
                                onClick={() => {
                                    handleAddNodeFromTemplate(template, 'labeling');
                                    handleClose();
                                }}
                                sx={{ display: 'flex', justifyContent: 'space-between' }}
                            >
                                <ListItemText>{template.name}</ListItemText>
                                {template.isDeletable && (
                                    <IconButton
                                        size="small"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            deleteLabelingTemplate(key);
                                            handleClose();
                                        }}
                                        aria-label={`delete ${template.name}`}
                                    >
                                        <DeleteIcon fontSize="small" />
                                    </IconButton>
                                )}
                            </MenuItem>
                        ))}
                    </NestedMenuItem>,
                    <MenuItem key="add-profiler" onClick={() => handleAddNode('dataProfiler')}>Data Profiler Node</MenuItem>,
                    <MenuItem key="add-scaling" onClick={() => handleAddNode('dataScaling')}>Data Scaling Node</MenuItem>,
                    <MenuItem key="add-adv-scaling" onClick={() => handleAddNode('advancedDataScaling')}>Advanced Data Scaling Node</MenuItem>,
                    <MenuItem key="add-validation" onClick={() => handleAddNode('dataValidation')}>Data Validation Node</MenuItem>,
                    <MenuItem key="add-charting" onClick={() => handleAddNode('charting')}>Charting Node</MenuItem>,
                    <MenuItem key="add-featcorr" onClick={() => handleAddNode('featuresCorrelation')}>Features Correlation Node</MenuItem>,
                    <MenuItem key="add-trainer" onClick={() => handleAddNode('modelTrainer')}>ML Trainer Node</MenuItem>,
                    <MenuItem key="add-neuralnet" onClick={() => handleAddNode('neuralNetworkTrainer')}>Neural Network Trainer Node</MenuItem>,
                    <MenuItem key="add-bagging" onClick={() => handleAddNode('baggingTrainer')}>Bagging Trainer Node</MenuItem>,
                    <MenuItem key="add-imbalance" onClick={() => handleAddNode('classImbalance')}>Class Imbalancer Node</MenuItem>,
                    <MenuItem key="add-hp-tuning" onClick={() => handleAddNode('hyperparameterTuning')}>Hyperparameter Tuning Node</MenuItem>,
                    <MenuItem key="add-predictor" onClick={() => handleAddNode('modelPredictor')}>ML Predictor Node</MenuItem>,
                    <MenuItem key="add-backtester" onClick={() => handleAddNode('backtester')}>Backtester Node</MenuItem>,
                ]}
                {/* --- MENU FOR EDITING EXISTING ELEMENTS --- */}
                {menu?.type === 'node' && (
                    // Use a fragment to group node-specific actions
                    [
                        <MenuItem key="rename" onClick={handleRenameNode}>
                            <EditIcon fontSize="small" sx={{ mr: 1.5 }} />
                            Rename
                        </MenuItem>,
                        <MenuItem key="delete" onClick={handleDeleteElement}>
                            <DeleteIcon fontSize="small" sx={{ mr: 1.5 }} />
                            Delete
                        </MenuItem>
                    ]
                )}
                {menu?.type === 'edge' && (
                    <MenuItem onClick={handleDeleteElement}>
                        <DeleteIcon fontSize="small" sx={{ mr: 1.5 }} />
                        Delete
                    </MenuItem>
                )}
                
                {/* --- SAVE OPTION FOR CUSTOM CODE NODES --- */}
                {menu?.type === 'node' && (nodes.find(n => n.id === menu.id)?.type === 'customCode' || nodes.find(n => n.id === menu.id)?.type === 'labeling') && (
                    <MenuItem onClick={handleSaveTemplate}>Save as Template</MenuItem>
                )}

            </Menu>
        </Box>
    );
};

export const PipelineEditor: React.FC = () => {
    return (
        <LocalizationProvider dateAdapter={AdapterDayjs}>
            <ReactFlowProvider>
                <PipelineEditorContent />
            </ReactFlowProvider>
        </LocalizationProvider>
    );
};
