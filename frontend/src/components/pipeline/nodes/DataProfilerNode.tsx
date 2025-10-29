// src/components/pipeline/nodes/DataProfilerNode.tsx

import React, { memo, useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import {
    Paper, Box, Select, MenuItem, FormControl, InputLabel,
    IconButton, CircularProgress, Stack, Typography
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import type { SelectChangeEvent } from '@mui/material/Select';
import { NodeHeader } from './NodeHeader';

interface DataProfilerNodeData {
    label: string;
    selectedFeature: string | null;
}

export const DataProfilerNode = memo(({ id, data }: NodeProps<DataProfilerNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId, pipelineNodeCache } = usePipeline();
    const amIProcessing = isProcessing && processingNodeId === id;

    // State to hold the list of features available for profiling
    const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);

    // This effect watches the cache for this node's output.
    // When the output appears (after running), it extracts the list of
    // feature columns sent from the backend to populate the dropdown.
    useEffect(() => {
        const nodeCache = pipelineNodeCache[id];
        if (nodeCache?.info?.feature_list) {
            setAvailableFeatures(nodeCache.info.feature_list);

            // If no feature is selected yet, select the first one automatically
            if (!data.selectedFeature && nodeCache.info.feature_list.length > 0) {
                updateNodeData(id, { selectedFeature: nodeCache.info.feature_list[0] });
            }
        }
    }, [pipelineNodeCache, id, data.selectedFeature, updateNodeData]);

    const handleFeatureChange = (event: SelectChangeEvent<string>) => {
        updateNodeData(id, { selectedFeature: event.target.value });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10,
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, background: '#555' }} />

            <NodeHeader nodeId={id} title={data.label} color="#28c519ff">
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

            <Box
                sx={{ p: 2.5 }}
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <Stack spacing={2}>
                    <Typography variant="body2" color="text.secondary">
                        Select a feature to analyze. Results will appear in the side panel.
                    </Typography>
                    <FormControl fullWidth size="small">
                        <InputLabel>Feature</InputLabel>
                        <Select
                            value={data.selectedFeature || ''}
                            label="Feature"
                            onChange={handleFeatureChange}
                            disabled={availableFeatures.length === 0}
                        >
                            {availableFeatures.length === 0 && (
                                <MenuItem disabled value="">
                                    <em>Run node to populate features</em>
                                </MenuItem>
                            )}
                            {availableFeatures.map((feature) => (
                                <MenuItem key={feature} value={feature}>
                                    {feature}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Stack>
            </Box>

            <Handle type="source" position={Position.Right} id="output" style={{ ...handleStyle, background: '#555' }} />
        </Paper>
    );
});