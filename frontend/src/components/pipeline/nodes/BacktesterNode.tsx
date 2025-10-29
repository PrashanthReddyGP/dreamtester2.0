import React, { useEffect, useRef } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, IconButton, CircularProgress, FormControl, Select, MenuItem, TextField, Typography, InputAdornment, InputLabel } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SaveIcon from '@mui/icons-material/Save';
import DeleteIcon from '@mui/icons-material/Delete';
import Editor from '@monaco-editor/react';
import { NodeHeader } from './NodeHeader';
import debounce from 'lodash.debounce';
import { usePipeline } from '../../../context/PipelineContext';
import type { BacktesterNodeData } from '../../../context/PipelineContext';
import LeaderboardIcon from '@mui/icons-material/Leaderboard';

const DEBOUNCE_DELAY = 500; // 500ms delay

export const BacktesterNode = ({ id, data }: NodeProps<BacktesterNodeData>) => {
    const { 
        updateNodeData, 
        executePipelineUpToNode, 
        isProcessing, 
        processingNodeId,
        backtestTemplates,
        saveBacktestTemplate,
        deleteBacktestTemplate,
        viewBacktestAnalysis
    } = usePipeline();

    const amIProcessing = isProcessing && processingNodeId === id;

    // // Create a ref to hold the latest data. This helps avoid stale closures in our debounced function.
    // const dataRef = useRef(data);
    // useEffect(() => {
    //     dataRef.current = data;
    // }, [data]);

    const handleConfigChange = (field: keyof BacktesterNodeData['config'], value: string | number) => {
        const isNumeric = ['initialCapital', 'riskPercent', 'rr', 'commission'].includes(field);
        const finalValue = isNumeric && typeof value === 'string' ? parseFloat(value) || 0 : value;
        updateNodeData(id, { config: { ...data.config, [field]: finalValue } });
    };
    
    // // This is our debounced function for code changes.
    // // We use useMemo to ensure the debounced function is not recreated on every render.
    // // The dependencies are stable, so this is created only once.
    // const debouncedCodeChange = React.useMemo(
    //     () => debounce((block: keyof BacktesterNodeData['codeBlocks'], code: string | undefined) => {
    //         // We use the data from the ref to ensure we have the latest state
    //         const currentCodeBlocks = dataRef.current.codeBlocks;
    //         updateNodeData(id, { codeBlocks: { ...currentCodeBlocks, [block]: code || '' } });
    //     }, DEBOUNCE_DELAY),
    //     [id, updateNodeData] // Stable dependencies
    // );

    // // This is the function we'll pass to the editor's onChange.
    // // It calls our debounced function.
    // const handleCodeChange = (block: keyof BacktesterNodeData['codeBlocks'], code: string | undefined) => {
    //     debouncedCodeChange(block, code);
    // };
    

    const handleCodeChange = (block: keyof BacktesterNodeData['codeBlocks'], code: string | undefined) => {
        updateNodeData(id, { codeBlocks: { ...data.codeBlocks, [block]: code || '' } });
    };
    
    // // Cleanup the debounced function on unmount to cancel any pending calls
    // useEffect(() => {
    //     return () => {
    //         debouncedCodeChange.cancel();
    //     };
    // }, [debouncedCodeChange]);

    const handleTemplateChange = (event: any) => {
        const templateKey = event.target.value;
        const selectedTemplate = backtestTemplates[templateKey];
        if (selectedTemplate) {
            try {
                const parsedData = JSON.parse(selectedTemplate.code);
                updateNodeData(id, { ...parsedData, selectedTemplateKey: templateKey });
            } catch (e) {
                console.error("Failed to parse backtest template", e);
            }
        }
    };

    const handleSave = async () => {
        try {
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
            const defaultTemplate = backtestTemplates['ma_crossover'];
            if (defaultTemplate) {
                const parsedData = JSON.parse(defaultTemplate.code);
                updateNodeData(id, { ...parsedData, selectedTemplateKey: 'ma_crossover' });
            }
        } catch (error) {
            console.error("Failed to delete template from node:", error);
        }
    };

    // --- Implement the analysis handler ---
    const handleAnalysis = () => {
        // This is now the only action needed.
        viewBacktestAnalysis(id);
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10,
    };
    
    const currentTemplate = data.selectedTemplateKey ? backtestTemplates[data.selectedTemplateKey] : null;
    const editorOptions = { 
        minimap: { enabled: false }, 
        fontSize: 13, 
        wordWrap: 'on' as const, 
        scrollBeyondLastLine: false, 
        lineNumbers: 'off' as const, 
        padding: { top: 8 }
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '500px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555' }} />

            <NodeHeader nodeId={id} title={data.label} color="#d72559ff">
                <Box>
                    <IconButton size="small" sx={{ color: 'white', mr: 1 }} onClick={handleAnalysis}>
                        <LeaderboardIcon fontSize="medium" />
                    </IconButton>
                    <IconButton size="small" sx={{ color: 'white' }} onClick={() => executePipelineUpToNode(id)} disabled={amIProcessing}>
                        {amIProcessing ? <CircularProgress size={24} color="inherit" /> : <PlayArrowIcon fontSize="medium" />}
                    </IconButton>
                </Box>
            </NodeHeader>
            
            <Box className="nodrag" sx={{ p: 1, borderBottom: '1px solid #444', display: 'flex', alignItems: 'center', gap: 1 }}>
                <FormControl fullWidth size="small">
                    <Select value={data.selectedTemplateKey || ''} onChange={handleTemplateChange} displayEmpty>
                        <MenuItem value="" disabled><em>Select a strategy...</em></MenuItem>
                        {Object.entries(backtestTemplates).map(([key, template]) => (
                            <MenuItem key={key} value={key}>{template.name}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <IconButton size="small" onClick={handleSave} aria-label="save"><SaveIcon /></IconButton>
                <IconButton size="small" onClick={handleDelete} disabled={!currentTemplate?.isDeletable} aria-label="delete"><DeleteIcon /></IconButton>
            </Box>
            
            <Box className="nodrag" sx={{ display: 'flex', flexDirection: 'column' }}>
                {/* CONFIGURATION SECTION */}
                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, p: 2, borderBottom: '1px solid #444' }}>
                    <TextField label="Initial Capital" type="number" size="small" value={data.config.initialCapital} onChange={(e) => handleConfigChange('initialCapital', e.target.value)} InputProps={{ startAdornment: <InputAdornment position="start">$</InputAdornment> }} />
                    <TextField label="Risk" type="number" size="small" value={data.config.riskPercent} onChange={(e) => handleConfigChange('riskPercent', e.target.value)} InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }} />
                    <TextField label="Reward/Risk Ratio" type="number" size="small" value={data.config.rr} onChange={(e) => handleConfigChange('rr', e.target.value)} />
                    <TextField label="Commission" type="number" size="small" value={data.config.commission} onChange={(e) => handleConfigChange('commission', e.target.value)} InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }} />
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
                            <MenuItem value="time_based" disabled>Time-Based</MenuItem>
                        </Select>
                    </FormControl>
                </Box>
                
                {/* CODE BLOCKS SECTION */}
                <Box sx={{ p: 1.5, borderBottom: '1px solid #444' }}>
                    <Typography variant="caption" sx={{ display: 'block', textAlign: "center", pb: 1 }}>ENTRY CONDITION</Typography>
                    <Editor height="80px" language="python" theme="app-dark-theme" value={data.codeBlocks.entryLogic} onChange={(v) => handleCodeChange('entryLogic', v)} options={editorOptions} />
                </Box>

                {data.config.exitType === 'tp_sl' ? (
                    <>
                        <Box sx={{ p: 1.5, borderBottom: '1px solid #444' }}>
                            <Typography variant="caption" sx={{ display: 'block', textAlign: "center", pb: 1 }}>STOP LOSS LOGIC</Typography>
                            <Editor height="60px" language="python" theme="app-dark-theme" value={data.codeBlocks.stopLossLogic} onChange={(v) => handleCodeChange('stopLossLogic', v)} options={editorOptions} />
                        </Box>
                        {/* <Box sx={{ p: 1.5 }}>
                            <Typography variant="caption" sx={{ display: 'block', textAlign: "center", pb: 0.5 }}>POSITION SIZING LOGIC</Typography>
                            <Editor height="80px" language="python" theme="app-dark-theme" value={data.codeBlocks.positionSizingLogic} onChange={(v) => handleCodeChange('positionSizingLogic', v)} options={editorOptions} />
                        </Box> */}
                    </>
                ) : (
                    <Box sx={{ p: 1.5 }}>
                        <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', pb: 1 }}>CUSTOM EXIT CONDITION</Typography>
                        <Editor height="100px" language="python" theme="app-dark-theme" value={data.codeBlocks.customExitLogic} onChange={(v) => handleCodeChange('customExitLogic', v)} options={editorOptions} />
                    </Box>
                )}
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};