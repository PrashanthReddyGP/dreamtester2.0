import React, { useMemo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, FormGroup, FormControlLabel, Checkbox, IconButton } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';

/**
 * Data structure for the ProcessIndicatorsNode.
 * `selectedIndicators` will store a map of { predecessorNodeId: isChecked }
 */
interface ProcessIndicatorsNodeData {
    label: string;
    selectedIndicators: { [nodeId: string]: boolean };
}

export const ProcessIndicatorsNode = ({ id, data }: NodeProps<ProcessIndicatorsNodeData>) => {
    // Access global state (nodes, edges, schema) from the context
    const { nodes, edges, indicatorSchema, updateNodeData } = usePipeline();

    // Memoized calculation to find connected predecessor feature nodes.
    // This will only re-run when the relevant dependencies (id, nodes, edges) change.
    const predecessorFeatures = useMemo(() => {
        // Find all edges that connect TO this node
        const parentIds = edges
            .filter(edge => edge.target === id)
            .map(edge => edge.source);

        // From the parent IDs, find the actual node objects that are of type 'feature'
        return nodes
            .filter(node => parentIds.includes(node.id) && node.type === 'feature')
            .map(node => ({
                id: node.id,
                indicatorKey: node.data.indicator,
                // Get the user-friendly name from the schema, or fall back to the key
                displayName: indicatorSchema[node.data.indicator]?.name || node.data.indicator || 'Unknown Indicator',
            }));
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
            <Handle type="target" position={Position.Left} style={handleStyle} isConnectable={true} />

            {/* Header */}
            <Box sx={{ bgcolor: '#41a336ff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1 }}>
                    {data.label || 'Process Indicators'}
                </Typography>
                <IconButton size="small" sx={{ color: 'white' }} aria-label="run">
                    <PlayArrowIcon />
                </IconButton>
            </Box>

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
            
            <Handle type="source" position={Position.Right} style={handleStyle} />
        </Paper>
    );
};