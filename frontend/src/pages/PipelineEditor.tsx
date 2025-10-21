// PipelineEditor.tsx
import React, { useState, useCallback, useRef, useMemo } from 'react';
import { Box, Menu, MenuItem, IconButton, ListItemText } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete'; // Import the delete icon
import ReactFlow, {
    Controls as ReactFlowControls, // Import with an alias
    Background,
    ReactFlowProvider,
    useReactFlow,
} from 'reactflow';
import type { Node, Edge } from 'reactflow'; // Add Node, Edge here
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

import EditIcon from '@mui/icons-material/Edit';
import { FeaturesCorrelationNode } from '../components/pipeline/nodes/FeaturesCorrelationNode';

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
        selectNode,
        selectedNodeId,
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
        editingNodeId,
        setEditingNodeId,
    } = usePipeline();

    const { project } = useReactFlow();

    const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);

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

    const handleRenameNode = () => {
        if (!menu || !menu.id || menu.type !== 'node') return;

        // 1. Set the editing state in the context.
        setEditingNodeId(menu.id);
        // 2. Set the selected state. This will change the `selected` prop
        //    on the node, busting the React.memo cache and forcing a re-render.
        selectNode(menu.id);

        handleClose();
    };

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
        customCode: EditorNode,
        customLabeling: LabelingNode,
        dataScaling: DataScalingNode,
        dataValidation: DataValidationNode,
        classImbalance: ClassImbalanceNode,
        charting: ChartingNode,
        featuresCorrelation: FeaturesCorrelationNode,
        modelTrainer: ModelTrainerNode,
        hyperparameterTuning: HyperparameterTuningNode, 
        modelPredictor: ModelPredictorNode,
        backtester: BacktesterNode,
    }), [symbolList, isFetchingSymbols, indicatorSchema, isLoadingSchema]); // Stable dependencies


    const handleDeleteElement = () => {
        if (!menu || !menu.id || (menu.type !== 'node' && menu.type !== 'edge')) return;
        handleClose();
        deleteElement(menu.id, menu.type);
    };

    const nodesWithControlledSelection = useMemo(() => {
        return nodes.map(node => ({
            ...node,
            selected: node.id === selectedNodeId,
        }));
    }, [nodes, selectedNodeId]);


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
        <Panel
            // The ref is no longer passed
            collapsible={true}
            collapsedSize={0}
            defaultSize={38}
            minSize={15}
            onCollapse={handleCollapse}
            onExpand={handleExpand}
        >
            <SidePanel
                isPanelOpen={isPanelOpen}
                panelPosition={panelPosition}
                setPanelPosition={setPanelPosition}
                togglePanel={togglePanel}
                displayData={dataForDisplay.data}
                displayInfo={dataForDisplay.info}
                selectedNode={nodes.find(n => n.id === selectedNodeId)} // Pass the selected node itself
            />
        </Panel>
    );

    const reactFlowComponent = (
        <Panel defaultSize={62} minSize={30}>
            <Box sx={{ width: '100%', height: '100%' }} ref={reactFlowWrapperRef}>
                <ReactFlow
                    nodes={nodesWithControlledSelection}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onConnectStart={onConnectStart}
                    onConnectEnd={onConnectEnd}
                    nodeTypes={nodeTypes}
                    onNodeClick={(_, node) => {
                        // Clicking another node should select it and exit rename mode
                        selectNode(node.id);
                        setEditingNodeId(null);
                    }}
                    onPaneClick={() => {
                        // Clicking the background should deselect and exit rename mode
                        selectNode(null);
                        setEditingNodeId(null);
                    }}
                    onPaneContextMenu={(e) => onContextMenu(e)}
                    onNodeContextMenu={(e, node) => onContextMenu(e, node)}
                    onEdgeContextMenu={(e, edge) => onContextMenu(e, edge)}
                    
                    elevateNodesOnSelect={true}
                    fitView
                    minZoom={0.2}
                    maxZoom={1}
                    panOnScroll={false}
                    selectionOnDrag={true}
                    panOnDrag={[1, 2]}
                    deleteKeyCode={['Delete', 'Backspace']}
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
                            nodes={nodesWithControlledSelection}
                            edges={edges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            onConnect={onConnect}
                            onConnectStart={onConnectStart}
                            onConnectEnd={onConnectEnd}
                            nodeTypes={nodeTypes}
                            onNodeClick={(_, node) => {
                                // Clicking another node should select it and exit rename mode
                                selectNode(node.id);
                                setEditingNodeId(null);
                            }}
                            onPaneClick={() => {
                                // Clicking the background should deselect and exit rename mode
                                selectNode(null);
                                setEditingNodeId(null);
                            }}
                            onPaneContextMenu={(e) => onContextMenu(e)}
                            onNodeContextMenu={(e, node) => onContextMenu(e, node)}
                            onEdgeContextMenu={(e, edge) => onContextMenu(e, edge)}
                            
                            elevateNodesOnSelect={true}
                            fitView
                            minZoom={0.2}
                            maxZoom={1.2}
                            panOnScroll={false}
                            selectionOnDrag={true}
                            panOnDrag={[1, 2]}
                            deleteKeyCode={['Delete', 'Backspace']}
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
                    <MenuItem key="add-scaling" onClick={() => handleAddNode('dataScaling')}>Add Data Scaling</MenuItem>,
                    <MenuItem key="add-validation" onClick={() => handleAddNode('dataValidation')}>Add Data Validation</MenuItem>,
                    <MenuItem key="add-charting" onClick={() => handleAddNode('charting')}>Add Charting Node</MenuItem>,
                    <MenuItem key="add-featcorr" onClick={() => handleAddNode('featuresCorrelation')}>Add Features Correlation Node</MenuItem>,
                    <MenuItem key="add-trainer" onClick={() => handleAddNode('modelTrainer')}>Add ML Trainer</MenuItem>,
                    <MenuItem key="add-imbalance" onClick={() => handleAddNode('classImbalance')}>Add Class Imbalancer</MenuItem>,
                    <MenuItem key="add-hp-tuning" onClick={() => handleAddNode('hyperparameterTuning')}>Add Hyperparameter Tuning</MenuItem>,
                    <MenuItem key="add-predictor" onClick={() => handleAddNode('modelPredictor')}>Add ML Predictor</MenuItem>,
                    <MenuItem key="add-backtester" onClick={() => handleAddNode('backtester')}>Add Backtester</MenuItem>,
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
