import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, IconButton, CircularProgress, FormControl, Select, MenuItem, TextField, Typography, InputAdornment, InputLabel } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SaveIcon from '@mui/icons-material/Save';
import DeleteIcon from '@mui/icons-material/Delete';
import Editor from '@monaco-editor/react';
import { NodeHeader } from './NodeHeader';

import { usePipeline } from '../../../context/PipelineContext';
import type { BacktesterNodeData } from '../../../context/PipelineContext'; // Import the new type

export const BacktesterNode = ({ id, data }: NodeProps<BacktesterNodeData>) => {
    const { 
        updateNodeData, 
        executePipelineUpToNode, 
        isProcessing, 
        processingNodeId,
        backtestTemplates,
        saveBacktestTemplate,
        deleteBacktestTemplate,
    } = usePipeline();

    const amIProcessing = isProcessing && processingNodeId === id;

    // Generic handler to update a config value
    const handleConfigChange = (field: keyof BacktesterNodeData['config'], value: any) => {
        const numericValue = typeof value === 'string' ? parseFloat(value) : value;
        updateNodeData(id, { config: { ...data.config, [field]: numericValue } });
    };
    
    // Generic handler to update a code block
    const handleCodeChange = (block: keyof BacktesterNodeData['codeBlocks'], code: string | undefined) => {
        updateNodeData(id, { codeBlocks: { ...data.codeBlocks, [block]: code || '' } });
    };

    const handleTemplateChange = (event: any) => {
        const templateKey = event.target.value;
        const selectedTemplate = backtestTemplates[templateKey];
        if (selectedTemplate) {
            try {
                // Templates store the entire data object as a stringified JSON
                const parsedData = JSON.parse(selectedTemplate.code);
                updateNodeData(id, { ...parsedData, selectedTemplateKey: templateKey });
            } catch (e) {
                console.error("Failed to parse backtest template", e);
            }
        }
    };

    const handleSave = async () => {
        try {
            // We stringify the entire data object to save it in the template's 'code' field
            const dataToSave = JSON.stringify({
                label: data.label,
                config: data.config,
                codeBlocks: data.codeBlocks
            });
            const newKey = await saveBacktestTemplate(data.label, 'Custom backtest strategy', dataToSave);
            updateNodeData(id, { selectedTemplateKey: newKey });
            alert(`Template "${data.label}" saved successfully!`);
        } catch (error) {
            console.error("Failed to save template:", error);
            if (!(error instanceof Error && error.message.includes("cancelled"))) {
                alert("Could not save template.");
            }
        }
    };

    const handleDelete = async () => {
        if (!data.selectedTemplateKey) return;
        try {
            await deleteBacktestTemplate(data.selectedTemplateKey);
            // Revert to the default template after deletion
            const defaultTemplate = backtestTemplates['ma_crossover'];
            if (defaultTemplate) {
                 const parsedData = JSON.parse(defaultTemplate.code);
                 updateNodeData(id, { ...parsedData, selectedTemplateKey: 'ma_crossover' });
            }
        } catch (error) {
            console.error("Failed to delete template from node:", error);
        }
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
    };
    
    const currentTemplate = data.selectedTemplateKey ? backtestTemplates[data.selectedTemplateKey] : null;

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '500px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555', zIndex: 10}} />

            <NodeHeader nodeId={id} title={data.label} color="#d72559ff">
                <IconButton size="small" sx={{ color: 'white' }} onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                    {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                </IconButton>
            </NodeHeader>
            
            <Box className="nodrag" sx={{ p: 1, borderBottom: '1px solid #444' }}>
                 {/* TEMPLATE CONTROLS */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <FormControl fullWidth size="small">
                        <Select value={data.selectedTemplateKey || ''} onChange={handleTemplateChange} displayEmpty sx={{ ml: '5px' }}>
                            <MenuItem value="" disabled><em>Select a strategy...</em></MenuItem>
                            {Object.entries(backtestTemplates).map(([key, template]) => (
                                <MenuItem key={key} value={key}>{template.name}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <IconButton size="small" sx={{ color: 'white' }} onClick={handleSave} aria-label="save"><SaveIcon /></IconButton>
                    <IconButton size="small" sx={{ color: 'white' }} onClick={handleDelete} disabled={!currentTemplate?.isDeletable} aria-label="delete"><DeleteIcon /></IconButton>
                </Box>
            </Box>
            
            <Box className="nodrag" sx={{ display: 'flex', flexDirection: 'column' }}>
                {/* CONFIGURATION SECTION */}
                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, p: 2 }}>
                    <TextField label="Initial Capital" type="number" size="small" value={data.config.initialCapital} onChange={(e) => handleConfigChange('initialCapital', e.target.value)} InputProps={{ startAdornment: <InputAdornment position="start">$</InputAdornment> }} />
                    <TextField label="Risk" type="number" size="small" value={data.config.riskPercent} onChange={(e) => handleConfigChange('riskPercent', e.target.value)} InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }} />
                    <TextField label="Reward/Risk Ratio" type="number" size="small" value={data.config.rr} onChange={(e) => handleConfigChange('rr', e.target.value)} />
                    <FormControl size="small">
                        <InputLabel>Trade Direction</InputLabel>
                        <Select value={data.config.tradeDirection} label="Trade Direction" onChange={(e) => handleConfigChange('tradeDirection', e.target.value)}>
                            <MenuItem value="hedge">Long & Short</MenuItem>
                            <MenuItem value="long">Long Only</MenuItem>
                            <MenuItem value="short">Short Only</MenuItem>
                        </Select>
                    </FormControl>
                    <FormControl size="small">
                         <InputLabel>Exit Type</InputLabel>
                        <Select value={data.config.exitType} label="Exit Type" onChange={(e) => handleConfigChange('exitType', e.target.value)}>
                            <MenuItem value="tp_sl">Take Profit / Stop Loss</MenuItem>
                            <MenuItem value="single_condition">Custom Exit Condition</MenuItem>
                            <MenuItem value="time_based" disabled>Time-Based (Coming Soon)</MenuItem>
                        </Select>
                    </FormControl>
                    <TextField label="Commission" type="number" size="small" value={data.config.riskPercent} onChange={(e) => handleConfigChange('riskPercent', e.target.value)} InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }} />
                </Box>
                
                {/* CODE BLOCKS SECTION */}
                <Box sx={{ mb: '10px', borderBottom: '1px solid #444' }}>
                    <Typography variant="caption" sx={{ mb: 0.5, display: 'block', textAlign: "center" }}>ENTRY LOGIC</Typography>
                    <Editor height="80px" language="python" theme="app-dark-theme" value={data.codeBlocks.indicators} onChange={(v) => handleCodeChange('indicators', v)} options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: 'on', scrollBeyondLastLine: false, lineNumbers: 'off' }} />
                </Box>
                <Box sx={{ borderBottom: '1px solid #444' }}>
                    <Typography variant="caption" sx={{ mb: 0.5, display: 'block', textAlign: "center" }}>EXIT LOGIC</Typography>
                     <Editor height="150px" language="python" theme="app-dark-theme" value={data.codeBlocks.entryLogic} onChange={(v) => handleCodeChange('entryLogic', v)} options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: 'on', scrollBeyondLastLine: false, lineNumbers: 'off' }} />
                </Box>

                {data.config.exitType === 'single_condition' && (
                    <Box>
                        <Typography variant="caption" sx={{ mb: 0.5, display: 'block', textAlign: 'center' }}>Custom Exit Logic (must return True to exit)</Typography>
                        <Editor height="120px" language="python" theme="app-dark-theme" value={data.codeBlocks.exitLogic} onChange={(v) => handleCodeChange('exitLogic', v)} options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: 'on', scrollBeyondLastLine: false, lineNumbers: 'off' }} />
                    </Box>
                )}

                <Box>
                    <Typography variant="caption" sx={{ mb: 0.5, display: 'block', textAlign: 'center' }}>POSITION SIZING</Typography>
                    <Editor height="80px" language="python" theme="app-dark-theme" value={data.codeBlocks.indicators} onChange={(v) => handleCodeChange('indicators', v)} options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: 'on', scrollBeyondLastLine: false, lineNumbers: 'off' }} />
                </Box>

            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};