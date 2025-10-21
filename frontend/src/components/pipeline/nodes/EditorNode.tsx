import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Box, IconButton, CircularProgress, FormControl, Select, MenuItem } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SaveIcon from '@mui/icons-material/Save';
import DeleteIcon from '@mui/icons-material/Delete';
import Editor from '@monaco-editor/react';
import type { OnChange } from '@monaco-editor/react';
import { NodeHeader } from './NodeHeader';

import { usePipeline } from '../../../context/PipelineContext';

interface EditorNodeData {
    label: string;
    code: string;
    selectedTemplateKey?: string; // Track which template is selected
}

export const EditorNode = ({ id, data }: NodeProps<EditorNodeData>) => {
    const { 
        updateNodeData, 
        executePipelineUpToNode, 
        isProcessing, 
        processingNodeId,
        feTemplates,
        saveFeTemplate,
        deleteFeTemplate,
    } = usePipeline();

    const amIProcessing = isProcessing && processingNodeId === id;

    const handleEditorChange: OnChange = (value) => {
        updateNodeData(id, { code: value || '' });
    };

    const handleTemplateChange = (event: any) => {
        const templateKey = event.target.value;
        const selectedTemplate = feTemplates[templateKey];
        if (selectedTemplate) {
            updateNodeData(id, { 
                label: selectedTemplate.name,
                code: selectedTemplate.code,
                selectedTemplateKey: templateKey 
            });
        }
    };

    const handleSave = async () => {
        try {
            const newKey = await saveFeTemplate(data.label, 'Custom template', data.code);
            updateNodeData(id, { selectedTemplateKey: newKey });
        } catch (error) {
            console.error("Failed to save template:", error);
            // Optionally, show an alert to the user if the save was cancelled
            if (error instanceof Error && error.message.includes("cancelled")) {
                 // alert("Save cancelled."); // This might be too noisy, up to you
            } else {
                alert("Could not save template.");
            }
        }
    };

    const handleDelete = async () => {
        if (!data.selectedTemplateKey) return;
        try {
            await deleteFeTemplate(data.selectedTemplateKey);
            const defaultTemplateKey = 'guide';
            const defaultTemplate = feTemplates[defaultTemplateKey];
            if (defaultTemplate) {
                updateNodeData(id, {
                    code: defaultTemplate.code,
                    label: defaultTemplate.name,
                    selectedTemplateKey: defaultTemplateKey
                });
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
        zIndex: 10
    };

    const currentTemplate = data.selectedTemplateKey ? feTemplates[data.selectedTemplateKey] : null;

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '450px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={{ ...handleStyle, backgroundColor: '#555'}} />

            <NodeHeader nodeId={id} title={data.label} color="#259fd7ff">
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

            {/* FIX: Add nodrag and stopPropagation to this container */}
            <Box 
                className="nodrag"
                onMouseDown={(e) => e.stopPropagation()}
                sx={{ p: 1, display: 'flex', alignItems: 'center', gap: 1, bgcolor: 'background.paper', borderBottom: '1px solid #444' }}
            >
                <FormControl fullWidth size="small">
                    <Select
                        value={data.selectedTemplateKey || ''}
                        onChange={handleTemplateChange}
                        displayEmpty
                        inputProps={{ 'aria-label': 'Select Template' }}
                        MenuProps={{ // This helps with positioning inside the ReactFlow pane
                            container: document.body 
                        }}
                        sx={{
                            bgcolor: '#2e2e2eff',
                            color: 'white',
                            '& .MuiSvgIcon-root': { color: 'white' },
                            '& .MuiOutlinedInput-notchedOutline': { border: 'none' }
                        }}
                    >
                        <MenuItem value="" disabled>
                            <em>Select a template...</em>
                        </MenuItem>
                        {Object.entries(feTemplates).map(([key, template]) => (
                            <MenuItem key={key} value={key}>
                                {template.name}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <IconButton size="small" sx={{ color: 'white' }} onClick={handleSave} aria-label="save template">
                    <SaveIcon />
                </IconButton>
                <IconButton
                    size="small"
                    sx={{ color: 'white' }}
                    onClick={handleDelete}
                    disabled={!currentTemplate?.isDeletable}
                    aria-label="delete template"
                >
                    <DeleteIcon />
                </IconButton>
            </Box>

            <Box
                className="nodrag"
                onMouseDown={(event) => event.stopPropagation()}
                sx={{
                    border: '1px solid #444',
                    borderRadius: '0 0 8px 8px',
                    overflow: 'hidden'
                }}
            >
                <Editor
                    height="200px"
                    language="python"
                    theme="app-dark-theme"
                    value={data.code}
                    onChange={handleEditorChange}
                    loading={<CircularProgress size={40} sx={{ display: 'block', margin: 'auto', my: 2 }} />}
                    options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        wordWrap: 'on',
                        scrollBeyondLastLine: false,
                        padding: {top: 10},
                        automaticLayout: true,
                        lineNumbers: 'off',
                    }}
                />
            </Box>
            
            <Handle type="source" position={Position.Right} style={{ ...handleStyle, backgroundColor: '#555'}} />
        </Paper>
    );
};