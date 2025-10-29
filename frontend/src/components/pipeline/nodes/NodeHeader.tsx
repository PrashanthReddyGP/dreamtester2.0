// src/components/pipeline/nodes/NodeHeader.tsx

import React, { useState, useEffect } from 'react';
import { Box, Typography, TextField } from '@mui/material';
import { usePipeline } from '../../../context/PipelineContext';

interface NodeHeaderProps {
    nodeId: string;
    title: string;
    color: string;
    textColor?: string; // <-- 1. Add an optional textColor prop
    children?: React.ReactNode; // For the run button or other icons
}

export const NodeHeader: React.FC<NodeHeaderProps> = ({ nodeId, title, color, textColor= 'white', children }) => {
    const { editingNodeId, setEditingNodeId, updateNodeData } = usePipeline();
    const amIEditing = editingNodeId === nodeId;

    const [tempLabel, setTempLabel] = useState(title);

    useEffect(() => {
        setTempLabel(title);
    }, [title]);

    const handleSave = () => {
        if (tempLabel.trim()) {
            updateNodeData(nodeId, { label: tempLabel.trim() });
        }
        setEditingNodeId(null);
    };

    const handleCancel = () => {
        setTempLabel(title);
        setEditingNodeId(null);
    };

    const handleKeyDown = (event: React.KeyboardEvent) => {
        if (event.key === 'Enter') {
            handleSave();
        } else if (event.key === 'Escape') {
            handleCancel();
        }
    };

    return (
        <Box
            className={amIEditing ? 'nodrag' : ''}
            onDoubleClick={() => {
                if (!amIEditing) {
                    setEditingNodeId(nodeId);
                }
            }}
            sx={{
                bgcolor: color,
                p: 1,
                borderTopLeftRadius: 'inherit',
                borderTopRightRadius: 'inherit',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                minHeight: '50px'
            }}
        >
            {amIEditing ? (
                <TextField
                    value={tempLabel}
                    onChange={(e) => setTempLabel(e.target.value)}
                    onBlur={handleSave}
                    onKeyDown={handleKeyDown}
                    size="small"
                    variant="standard"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                    sx={{
                        flexGrow: 1,
                        mr: 1,
                        '& .MuiInput-underline:before': { borderBottomColor: 'rgba(255,255,255,0.7)' },
                        '& .MuiInput-underline:hover:not(.Mui-disabled):before': { borderBottomColor: 'white' },
                        input: { color: 'white', fontWeight: '500' },
                    }}
                />
            ) : (
                <Typography 
                    variant="subtitle2" 
                    sx={{ color: textColor, fontWeight: 'bold', pl: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', userSelect: 'none' }}
                    >
                    {(title || '').toUpperCase()} 
                </Typography>
            )}
            {children}
        </Box>
    );
};