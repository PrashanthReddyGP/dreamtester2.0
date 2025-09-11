// PipelineEditor.tsx
import React, { useState, useCallback, useRef, useMemo } from 'react';
import { Box, Menu, MenuItem } from '@mui/material';
import ReactFlow, {
    Controls as ReactFlowControls, // Import with an alias
    Background,
    ReactFlowProvider,
    useReactFlow,
} from 'reactflow';
import type {
    Node,
    Edge,
} from 'reactflow'
import 'reactflow/dist/style.css';
import { styled } from '@mui/material/styles'; // Import styled

import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';

import { DataSourceNode } from '../components/pipeline/nodes/DataSourceNode';
import { FeatureNode } from '../components/pipeline/nodes/FeatureNode';
import { usePipeline } from '../context/PipelineContext';

import {
    Panel,
    PanelGroup,
    PanelResizeHandle,
} from "react-resizable-panels";
import { SidePanel, CollapsedPanelHandle } from '../components/pipeline/SidePanel';
import { ProcessIndicatorsNode } from '../components/pipeline/nodes/ProcessIndicatorsNode';
import { EditorNode } from '../components/pipeline/nodes/EditorNode';
import { MLModelNode } from '../components/pipeline/nodes/MLModelNode';

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
    } = usePipeline();

    const { project } = useReactFlow();

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
    const [panelPosition, setPanelPosition] = useState<'left' | 'right' | 'top' | 'bottom'>('bottom');

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

    const handleDeleteElement = () => {
        if (!menu || !menu.id || (menu.type !== 'node' && menu.type !== 'edge')) return;
        handleClose();
        deleteElement(menu.id, menu.type);
    };

    // This `useMemo` now correctly depends on data from the context.
    // It will re-evaluate when the data finishes loading.
    const nodeTypes = useMemo(() => ({
        dataSource: (props: any) => (
            <DataSourceNode
                {...props}
                symbolList={symbolList}
                isFetchingSymbols={isFetchingSymbols}
                onFetch={handleFetchData}
                isFetching={isFetching}
            />
        ),
        feature: (props: any) => (
            <FeatureNode
                {...props}
                indicatorSchema={indicatorSchema}
                isLoadingSchema={isLoadingSchema}
            />
        ),
        processIndicators: ProcessIndicatorsNode,
        customCode: EditorNode,
        mlModel: MLModelNode,
    }), [symbolList, isFetchingSymbols, indicatorSchema, isLoadingSchema]);
    
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
            defaultSize={45}
            minSize={15}
            onCollapse={handleCollapse}
            onExpand={handleExpand}
        >
            <SidePanel
                isPanelOpen={isPanelOpen}
                panelPosition={panelPosition}
                setPanelPosition={setPanelPosition}
                togglePanel={togglePanel}
                displayData={displayData.data}
                displayInfo={displayData.info}
            />
        </Panel>
    );

    const reactFlowComponent = (
        <Panel defaultSize={75} minSize={30}>
            <Box sx={{ width: '100%', height: '100%' }} ref={reactFlowWrapperRef}>
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
        <Box sx={{ width: '100vw', height: 'calc(100vh - 85px)', position: 'relative' }}>
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
                    <MenuItem key="add-code" onClick={() => handleAddNode('customCode')}>Add Custom Code</MenuItem>,
                    <MenuItem key="add-ml" onClick={() => handleAddNode('mlModel')}>Add ML Model</MenuItem>,
                ]}
                {(menu?.type === 'node' || menu?.type === 'edge') && (
                    <MenuItem onClick={handleDeleteElement}>Delete</MenuItem>
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