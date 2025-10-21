import React, { useMemo } from 'react';
import { Handle, Position, getIncomers } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, FormGroup, FormControlLabel, Checkbox, IconButton, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader'; // Import the new header

/**
 * Data structure for the ProcessIndicatorsNode.
 * `selectedIndicators` will store a map of { predecessorNodeId: isChecked }
 */
interface ProcessIndicatorsNodeData {
    label: string;
    selectedIndicators: { [nodeId: string]: boolean };
}

export const ProcessIndicatorsNode = ({ id, data }: NodeProps<ProcessIndicatorsNodeData>) => {
    // 1. Pull from context
    const { nodes, edges, indicatorSchema, updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    
    // 2. Determine processing state
    const amIProcessing = isProcessing && processingNodeId === id;

    // Memoized calculation to find connected predecessor feature nodes.
    // This is now much cleaner and more reliable using getIncomers.
    const predecessorFeatures = useMemo(() => {
        // Find the current node object from the full list of nodes
        const currentNode = nodes.find(node => node.id === id);
        if (!currentNode) {
            return [];
        }

        // Use React Flow's getIncomers helper to get all direct parent nodes
        const incomers = getIncomers(currentNode, nodes, edges);
        
        // Filter the parents to only include nodes of type 'feature'
        return incomers
            .filter(node => node.type === 'feature')
            .map(node => ({
                id: node.id,
                indicatorKey: node.data.name,
                // Get the user-friendly name from the schema, or fall back to the key
                displayName: indicatorSchema[node.data.name]?.name || node.data.name || 'Unknown Indicator',
            }));
    // The dependencies remain the same, as we still rely on the same source data
    }, [id, nodes, edges, indicatorSchema]);

    // Handler for when a checkbox is toggled
    const handleCheckChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name: nodeId, checked } = event.target;
        
        // Create the new state for the `selectedIndicators` object
        const updatedSelection = {
            ...(data.selectedIndicators || {}), // Safely access previous state
            [nodeId]: checked,
        };
        
        // Update the node's data in the global context
        updateNodeData(id, { selectedIndicators: updatedSelection });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555' }} isConnectable={true} />

            {/* Header */}

            <NodeHeader nodeId={id} title={data.label} color="#41a336ff">
                <IconButton 
                    size="small" 
                    sx={{ color: 'white' }} 
                    aria-label="run" 
                    onClick={() => executePipelineUpToNode(id)}
                    disabled={amIProcessing}
                >
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>

            {/* <Box sx={{ bgcolor: '#41a336ff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1 }}>
                    {data.label || 'Process Indicators'}
                </Typography>
                <IconButton 
                    size="small" 
                    sx={{ color: 'white' }} 
                    aria-label="run"
                    onClick={() => executePipelineUpToNode(id)}
                    disabled={amIProcessing}
                >
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon />}
                </IconButton>
            </Box> */}

            {/* Content Body */}
            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                {/* <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1 }}>
                    Select indicators to process:
                </Typography> */}
                <FormGroup>
                    {predecessorFeatures.length > 0 ? (
                        predecessorFeatures.map(feature => (
                            <FormControlLabel
                                key={feature.id}
                                control={
                                    <Checkbox
                                        // Use `!!` to ensure it's a boolean, handles undefined case
                                        checked={!!data.selectedIndicators?.[feature.id]}
                                        onChange={handleCheckChange}
                                        name={feature.id} // The checkbox name is the predecessor node's ID
                                    />
                                }
                                label={feature.displayName}
                                sx={{textOverflow: 'ellipsis', whiteSpace: 'wrap', overflow: 'hidden'}}
                            />
                        ))
                    ) : (
                        <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic', textAlign: 'center' }}>
                            Connect Feature nodes to select.
                        </Typography>
                    )}
                </FormGroup>
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};