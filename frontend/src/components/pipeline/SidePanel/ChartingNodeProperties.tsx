// src/components/pipeline/SidePanel/ChartingNodeProperties.tsx

import React, { useMemo } from 'react';
import type { Node } from 'reactflow';
import { Stack, FormControl, InputLabel, Select, MenuItem, Typography, Button, Divider } from '@mui/material';
import { usePipeline } from '../../../context/PipelineContext';

interface ChartingNodePropertiesProps {
    node: Node;
}

export const ChartingNodeProperties: React.FC<ChartingNodePropertiesProps> = ({ node }) => {
    const { updateNodeData, pipelineNodeCache, executePipelineUpToNode } = usePipeline();
    const { data } = node;

    // Get the data from the *parent* of the charting node to populate the dropdowns
    const parentData = pipelineNodeCache[node.id] || { data: [], info: null };
    const availableColumns = useMemo(() => {
        if (parentData.data.length > 0) {
            return Object.keys(parentData.data[0]);
        }
        return [];
    }, [parentData.data]);

    const handleConfigChange = (field: string, value: any) => {
        updateNodeData(node.id, { [field]: value });
    };

    return (
        <Stack spacing={2.5} sx={{ p: 2 }}>
            <Typography variant="h6">Charting Configuration</Typography>
            <Divider />

            <FormControl fullWidth size="small">
                <InputLabel>Chart Type</InputLabel>
                <Select
                    value={data.chartType || 'scatter'}
                    label="Chart Type"
                    onChange={(e) => handleConfigChange('chartType', e.target.value)}
                >
                    <MenuItem value="scatter">Scatter Plot</MenuItem>
                    <MenuItem value="line">Line Chart</MenuItem>
                    <MenuItem value="histogram">Histogram</MenuItem>
                </Select>
            </FormControl>

            <FormControl fullWidth size="small">
                <InputLabel>X-Axis</InputLabel>
                <Select
                    value={data.xAxis || ''}
                    label="X-Axis"
                    onChange={(e) => handleConfigChange('xAxis', e.target.value)}
                >
                    {availableColumns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                </Select>
            </FormControl>

            {data.chartType !== 'histogram' && (
                 <FormControl fullWidth size="small">
                    <InputLabel>Y-Axis</InputLabel>
                    <Select
                        value={data.yAxis || ''}
                        label="Y-Axis"
                        onChange={(e) => handleConfigChange('yAxis', e.target.value)}
                    >
                        {availableColumns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                    </Select>
                </FormControl>
            )}

            {data.chartType === 'scatter' && (
                <FormControl fullWidth size="small">
                    <InputLabel>Group By (Color)</InputLabel>
                    <Select
                        value={data.groupBy || ''}
                        label="Group By (Color)"
                        onChange={(e) => handleConfigChange('groupBy', e.target.value)}
                    >
                         <MenuItem value=""><em>None</em></MenuItem>
                        {availableColumns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                    </Select>
                </FormControl>
            )}

            <Button variant="contained" onClick={() => executePipelineUpToNode(node.id)}>
                Generate Chart
            </Button>
        </Stack>
    );
};