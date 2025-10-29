// src/components/pipeline/nodes/LoopNode.tsx
import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, TextField, FormControl, InputLabel, Select, MenuItem, Typography } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import { NodeHeader } from './NodeHeader';
import { usePipeline } from '../../../context/PipelineContext';
import { StopPropagationBox } from './StopPropagationBox';

// Define the structure of the data for our Loop Node
export interface LoopNodeData {
    label: string;
    variableName: string;
    loopType: 'numeric_range' | 'value_list';
    numericStart: number;
    numericEnd: number;
    numericStep: number;
    valueList: string; // Comma-separated string
}

export const LoopNode = ({ id, data }: NodeProps<LoopNodeData>) => {
    const { updateNodeData } = usePipeline();

    const handleFieldChange = (fieldName: string, value: any) => {
        updateNodeData(id, { [fieldName]: value });
    };

    const handleSelectChange = (event: SelectChangeEvent<string>) => {
        handleFieldChange(event.target.name, event.target.value);
    };

    const handleTextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        handleFieldChange(event.target.name, event.target.value);
    };

    const handleNumericChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        handleFieldChange(name, parseFloat(value) || 0);
    };
    
    const handleStyle = { width: 12, height: 12, border: '1px solid #555', zIndex: 10 };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '300px', border: '1px solid #555' }}>
            {/* --- INPUT HANDLES --- */}

            <Handle
                type="target"
                position={Position.Left}
                id="loop-back" // Signal from the end of the loop body
                style={{ ...handleStyle, top: '25%', background: '#FFC107' }}
            />
            <Typography variant="caption" sx={{ position: 'absolute', left: -70, top: '25%', transform: 'translateY(-50%)', color: '#ccc' }}>
                Loop Back
            </Typography>

            <Handle
                type="target"
                position={Position.Left}
                id="data-in" // Data from the main pipeline
                style={{ ...handleStyle, top: '75%', background: '#4CAF50' }}
            />
            <Typography variant="caption" sx={{ position: 'absolute', left: -70, top: '75%', transform: 'translateY(-50%)', color: '#ccc' }}>
                Data Input
            </Typography>

            <NodeHeader nodeId={id} title={data.label} color="#d75b25" />

            <StopPropagationBox>

                <Box
                    className="nodrag"
                    onMouseDown={(e) => e.stopPropagation()}
                    sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                >
                    <TextField label="Variable Name" name="variableName" value={data.variableName} onChange={handleTextChange} size="small" fullWidth />
                    <FormControl fullWidth size="small">
                        <InputLabel id="loop-type-label">Loop Type</InputLabel>
                        <Select labelId="loop-type-label" name="loopType" value={data.loopType} label="Loop Type" onChange={handleSelectChange}>
                            <MenuItem value="numeric_range">Numeric Range</MenuItem>
                            <MenuItem value="value_list">List of Values</MenuItem>
                        </Select>
                    </FormControl>

                    {data.loopType === 'numeric_range' && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
                            <TextField label="Start" name="numericStart" type="number" value={data.numericStart} onChange={handleNumericChange} size="small" />
                            <TextField label="End" name="numericEnd" type="number" value={data.numericEnd} onChange={handleNumericChange} size="small" />
                            <TextField label="Step" name="numericStep" type="number" value={data.numericStep} onChange={handleNumericChange} size="small" />
                        </Box>
                    )}

                    {data.loopType === 'value_list' && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
                            <TextField label="Values (comma-separated)" name="valueList" value={data.valueList} onChange={handleTextChange} size="small" helperText="e.g., 10,20,50,100" />
                        </Box>
                    )}
                </Box>
                
            </StopPropagationBox>

             {/* --- OUTPUT HANDLES --- */}
            <Handle
                type="source"
                position={Position.Right}
                id="loop-body" // To the nodes inside the loop
                style={{ ...handleStyle, top: '25%', background: '#2196F3' }}
            />
            <Typography variant="caption" sx={{ position: 'absolute', right: -75, top: '25%', transform: 'translateY(-50%)', color: '#ccc' }}>
                Loop Body
            </Typography>

            <Handle
                type="source"
                position={Position.Right}
                id="loop-end" // To the rest of the pipeline
                style={{ ...handleStyle, top: '75%', background: '#F44336' }}
            />
            <Typography variant="caption" sx={{ position: 'absolute', right: -70, top: '75%', transform: 'translateY(-50%)', color: '#ccc' }}>
                Loop End
            </Typography>
        </Paper>
    );
};