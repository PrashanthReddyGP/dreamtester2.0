// src/components/pipeline/nodes/AdvancedDataScalingNode.tsx

import React, { memo, useState, useEffect, useMemo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import {
    Paper, Box, IconButton, CircularProgress, Stack, Typography,
    Autocomplete, TextField
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader';

interface AdvancedDataScalingNodeData {
    label: string;
    standardFeatures: string[];
    minmaxFeatures: string[];
    robustFeatures: string[];
    isConfigured: boolean;
}

export const AdvancedDataScalingNode = memo(({ id, data }: NodeProps<AdvancedDataScalingNodeData>) => {
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId, pipelineNodeCache } = usePipeline();
    const amIProcessing = isProcessing && processingNodeId === id;

    const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);

    useEffect(() => {
        const nodeCache = pipelineNodeCache[id];
        if (nodeCache?.info?.feature_list) {
            setAvailableFeatures(nodeCache.info.feature_list);
            // If the node is not yet configured, this means the discovery
            // run just finished. We now permanently mark it as configured.
            if (!data.isConfigured) {
                updateNodeData(id, { isConfigured: true });
            }
        }
    }, [pipelineNodeCache, id, data.isConfigured, updateNodeData]);

    const handleSelectionChange = (field: keyof Omit<AdvancedDataScalingNodeData, 'isConfigured' | 'label'>, value: string[]) => {
        updateNodeData(id, { [field]: value });
    };

    // Memoize the available options for each dropdown to prevent re-selection issues
    const selectableOptions = useMemo(() => {
        const selected = new Set([
            ...(data.standardFeatures || []),
            ...(data.minmaxFeatures || []),
            ...(data.robustFeatures || []),
        ]);
        return availableFeatures.filter(f => !selected.has(f));
    }, [availableFeatures, data.standardFeatures, data.minmaxFeatures, data.robustFeatures]);
    
    // Use the isConfigured flag to determine what to show
    const showConfigurationUI = data.isConfigured && availableFeatures.length > 0;

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10,
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '350px', border: '1px solid #555' }}>
            {/* Input Handles */}
            <Handle type="target" position={Position.Left} id="train_in" style={{ ...handleStyle, top: '33%', background: '#f44336' }} />
            <Handle type="target" position={Position.Left} id="test_in" style={{ ...handleStyle, top: '66%', background: '#4caf50' }} />

            <NodeHeader nodeId={id} title={data.label} color="#bb2b4aff">
                <IconButton size="small" sx={{ color: 'white' }} aria-label="run" onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>

            <Box sx={{ p: 2.5 }} className="nodrag" onMouseDown={(e) => e.stopPropagation()}>
                {showConfigurationUI ? (
                    <Stack spacing={2}>
                        <Autocomplete
                            multiple
                            size="small"
                            options={[...selectableOptions, ...(data.standardFeatures || [])]}
                            value={data.standardFeatures || []}
                            onChange={(_, newValue) => handleSelectionChange('standardFeatures', newValue)}
                            renderInput={(params) => <TextField {...params} label="Standard Scaler Features" />}
                        />
                        <Autocomplete
                            multiple
                            size="small"
                            options={[...selectableOptions, ...(data.minmaxFeatures || [])]}
                            value={data.minmaxFeatures || []}
                            onChange={(_, newValue) => handleSelectionChange('minmaxFeatures', newValue)}
                            renderInput={(params) => <TextField {...params} label="Min-Max Scaler Features" />}
                        />
                        <Autocomplete
                            multiple
                            size="small"
                            options={[...selectableOptions, ...(data.robustFeatures || [])]}
                            value={data.robustFeatures || []}
                            onChange={(_, newValue) => handleSelectionChange('robustFeatures', newValue)}
                            renderInput={(params) => <TextField {...params} label="Robust Scaler Features" />}
                        />
                    </Stack>
                ) : (
                    <Typography variant="body2" color="text.secondary" textAlign="center">
                        Connect and run the node to detect and assign features.
                    </Typography>
                )}
            </Box>

            {/* Output Handles */}
            <Handle type="source" position={Position.Right} id="train_out" style={{ ...handleStyle, top: '33%', background: '#f44336' }} />
            <Handle type="source" position={Position.Right} id="test_out" style={{ ...handleStyle, top: '66%', background: '#4caf50' }} />
        </Paper>
    );
});