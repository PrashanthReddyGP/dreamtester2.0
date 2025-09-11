import React from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Paper, Typography, Box, IconButton, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import Editor from '@monaco-editor/react';
import type { OnChange } from '@monaco-editor/react';

import { usePipeline } from '../../../context/PipelineContext';

// Data structure remains the same
interface EditorNodeData {
    label: string;
    code: string;
}

export const EditorNode = ({ id, data }: NodeProps<EditorNodeData>) => {
    const { updateNodeData } = usePipeline();

    // The onChange handler for Monaco has a different signature
    const handleEditorChange: OnChange = (value) => {
        // value can be undefined, so we provide a fallback
        updateNodeData(id, { code: value || '' });
    };

    const handleStyle = {
        width: 12,
        height: 12,
        background: 'primary.main',
        border: '1px solid #555',
        zIndex: 10
    };

    return (
        <Paper elevation={3} sx={{ borderRadius: 2, width: '450px', border: '1px solid #555' }}>
            <Handle type="target" position={Position.Left} style={handleStyle} />

            {/* Header */}
            <Box sx={{ bgcolor: '#259fd7ff', p: 1, borderTopLeftRadius: 'inherit', borderTopRightRadius: 'inherit', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.contrastText', pl: 1 }}>
                    {data.label || 'Custom Code'}
                </Typography>
                <IconButton size="small" sx={{ color: 'white' }} aria-label="run">
                    <PlayArrowIcon />
                </IconButton>
            </Box>

            {/* Monaco Editor Body */}
            <Box
                className="nodrag" // Prevents dragging the whole node when interacting with the editor
                onMouseDown={(event) => event.stopPropagation()}
                sx={{
                    // Add a border and slight padding for visual separation
                    border: '1px solid #444',
                    borderRadius: '0 0 8px 8px', // Match paper's bottom radius
                    overflow: 'hidden' // Important for border-radius to apply to the editor
                }}
            >
                <Editor
                    height="200px" // Set a fixed height
                    language="python"
                    theme="app-dark-theme" // A nice dark theme
                    value={data.code}
                    onChange={handleEditorChange}
                    loading={<CircularProgress size={40} sx={{ display: 'block', margin: 'auto', my: 2 }} />} // Show a spinner while loading
                    options={{
                        minimap: { enabled: false }, // Disable the minimap for a cleaner look
                        fontSize: 14,
                        wordWrap: 'on',
                        scrollBeyondLastLine: false,
                        padding: {top: 10},
                        automaticLayout: true, // Helps with resizing
                        lineNumbers: 'off',
                    }}
                />
            </Box>
            
            <Handle type="source" position={Position.Right} style={handleStyle} />
        </Paper>
    );
};