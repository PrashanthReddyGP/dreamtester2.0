import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { Node, NodeProps } from 'reactflow';
import { Paper, Box, Select, MenuItem, FormControl, InputLabel, IconButton, Autocomplete, TextField, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import type { SelectChangeEvent } from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import dayjs, { Dayjs } from 'dayjs'; // Import dayjs for date manipulation
import { usePipeline } from '../../../context/PipelineContext';
import { NodeHeader } from './NodeHeader'; // Import the new header

// Define the available timeframes in a constant
const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];

interface DataSourceNodeData {
    label: string;
    symbol: string;
    timeframe: string;
    startDate: string;
    endDate: string;
    setNodes: React.Dispatch<React.SetStateAction<Node<any>[]>>;
    reactFlowWrapperRef: React.RefObject<HTMLDivElement>;
}

// This defines the full props object the component receives,
// including the dynamic data passed from PipelineEditor
interface DataSourceNodeProps extends NodeProps<DataSourceNodeData> {
    symbolList: string[];
    isFetchingSymbols: boolean;
}

export const DataSourceNode = memo(({ id, data, symbolList, isFetchingSymbols}: DataSourceNodeProps) => {
    // 1. Pull everything needed from the context
    const { updateNodeData, executePipelineUpToNode, isProcessing, processingNodeId } = usePipeline();
    
    // 2. The component itself determines if it's the one being processed
    const amIProcessing = isProcessing && processingNodeId === id;

    // A single, generic handler for all field changes
    const handleFieldChange = (fieldName: string, value: any) => {
        updateNodeData(id, { [fieldName]: value });
    };

    const handleInputChange = (event: SelectChangeEvent<string>) => {
        const { name, value } = event.target;

        // Use the context-aware update function
        handleFieldChange(name, value); 
        };

    const handleDateChange = (newValue: Dayjs | null, fieldName: 'startDate' | 'endDate') => {
        if (!newValue) return;
        
        // Format the date back to a 'YYYY-MM-DD' string before saving to state
        const formattedDate = newValue.format('YYYY-MM-DD');

        // Use the context-aware update function
        handleFieldChange(fieldName, formattedDate);
    };

    // The MenuProps is still a good idea for future-proofing against z-index issues,
    // even though it wasn't the root cause here.
    const menuProps = {
        container: data.reactFlowWrapperRef?.current,
        style: {
            // Position the menu correctly relative to the select input
            position: 'absolute' as const,
        },
        // We can also increase the z-index here as an extra precaution
        PaperProps: {
            style: {
                zIndex: 1500 // Ensure it's higher than React Flow's controls (usually ~1000)
            }
        }
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main', // Match the edge color
        border: '1px solid #555',
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '250px', border: '1px solid #555' }}>

            <NodeHeader nodeId={id} title={data.label} color="#9820adff">
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

            {/* <Box sx={{ bgcolor: '#9820adff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignContent: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1, alignSelf: 'center' }}>
                    {data.label}
                </Typography>
                <IconButton 
                    size="small" 
                    sx={{ color: 'white' }} 
                    aria-label="run" 
                    onClick={() => executePipelineUpToNode(id)} // Use the function from context
                    disabled={amIProcessing} // Use the calculated boolean
                >
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </Box> */}

            <Box
                sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}
                // Add the 'nodrag' class to tell React Flow to ignore drag events on this element.
                className="nodrag"
                // As an extra, more forceful measure, stop the mousedown event from bubbling.
                onMouseDown={(event) => event.stopPropagation()}
            >
                {/* Symbol Dropdown */}
                <Autocomplete
                    options={symbolList}
                    value={data.symbol || null}
                    onChange={(event, newValue) => {
                        handleFieldChange('symbol', newValue || '');
                    }}
                    loading={isFetchingSymbols}
                    size="small"
                    slotProps={{
                        paper: {
                            sx: {
                                width: '225px',
                            },
                        },
                    }}
                    renderInput={(params) => (
                        <TextField
                            {...params}
                            label="Symbol"
                            InputProps={{
                                ...params.InputProps,
                                endAdornment: (
                                    <>
                                        {isFetchingSymbols ? <CircularProgress color="inherit" size={20} /> : null}
                                        {params.InputProps.endAdornment}
                                    </>
                                ),
                            }}
                        />
                    )}
                />
                {/* Timeframe Dropdown */}
                <FormControl fullWidth size="small">
                    <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
                    <Select
                        labelId="timeframe-select-label"
                        name="timeframe"
                        value={data.timeframe}
                        label="Timeframe"
                        onChange={handleInputChange}
                        MenuProps={menuProps}
                    >
                        {TIMEFRAMES.map((tf) => (
                            <MenuItem key={tf} value={tf}>{tf}</MenuItem>
                        ))}
                    </Select>
                </FormControl>

                <DatePicker
                    label="Start Date"
                    // The DatePicker value needs to be a dayjs object
                    value={dayjs(data.startDate)}
                    onChange={(newValue) => handleDateChange(newValue, 'startDate')}
                    // Use slotProps to style the underlying TextField
                    slotProps={{ textField: { size: 'small', fullWidth: true } }}
                    maxDate={dayjs(data.endDate)}
                />
                <DatePicker
                    label="End Date"
                    value={dayjs(data.endDate)}
                    onChange={(newValue) => handleDateChange(newValue, 'endDate')}
                    slotProps={{ textField: { size: 'small', fullWidth: true } }}
                    minDate={dayjs(data.startDate)}
                    maxDate={dayjs('2200-01-01')}
                />
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
});