// src/components/pipeline/nodes/NotesNode.tsx

import React, { memo } from 'react';
import type { NodeProps } from 'reactflow';
import { Paper, TextField, Box } from '@mui/material';
import StickyNote2Icon from '@mui/icons-material/StickyNote2'; // A fitting icon for notes
import { NodeHeader } from './NodeHeader'; // Import the new header
import { usePipeline } from '../../../context/PipelineContext'; // 1. Import the context hook
import { NodeResizer } from '@reactflow/node-resizer';
import '@reactflow/node-resizer/dist/style.css';

interface NotesNodeData {
    label: string;
    noteContent: string;
}

export const NotesNode = memo(({ id, data, selected }: NodeProps<NotesNodeData>) => {

    // 2. Get the function to update node data from the context
    const { updateNodeData } = usePipeline();

    // 3. Create a handler to update the node's data when the text changes
    const handleNoteChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        // Update the noteContent property in this node's data object
        updateNodeData(id, { ...data, noteContent: event.target.value });
    };

    return (
        <Paper 
            elevation={selected ? 6 : 3} 
            sx={{ 
                borderRadius: 2, 
                width: '100%',
                height: '100%',
                border: selected ? '2px solid #ffc107' : '1px solid #555', // A yellow/amber color for selection
                transition: 'border 0.2s ease-in-out',
                backgroundColor: '#fffbe0', // A light yellow background, like a sticky note
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {/* 3. Add the NodeResizer component */}
            <NodeResizer 
                isVisible={selected} 
                minWidth={150} 
                minHeight={100} 
            />

            <NodeHeader nodeId={id} title={data.label || "Note"} color="#eebc26ff" textColor='black'>
                <StickyNote2Icon sx={{ color: 'black' }} />
            </NodeHeader>

            {/* This box will grow to fill the remaining space */}
            <Box sx={{ p: 2, pt: 1, flexGrow: 1, height: '100%' }}>
                {/* 4. Replace Typography with an editable TextField */}
                <TextField
                    fullWidth
                    multiline
                    variant="standard" // Use "standard" variant for a clean look
                    placeholder="Type your notes here..."
                    value={data.noteContent || ''} // Control the component with data from React Flow
                    onChange={handleNoteChange} // Connect the change handler
                    // Remove the default underline and customize styling to make it look like plain text
                    InputProps={{
                        disableUnderline: true,
                        style: {
                           fontSize: '1.1rem', // Match the 'body2' typography variant
                            lineHeight: '1.4',
                            color: 'black',
                           height: '100%' // Ensure input takes full height
                        }
                    }}
                    // Ensure the input fills the box without extra padding
                    sx={{
                        height: '100%', // Make TextField fill the parent Box
                        '& .MuiInputBase-root': {
                            padding: 0,
                            height: '100%', // Ensure the root of the input takes full height
                        },
                        // Ensure the textarea itself fills its container
                        '& .MuiInputBase-inputMultiline': {
                            height: '100% !important',
                            overflowY: 'auto !important'
                        }
                    }}
                />
            </Box>

            
            {/* This node is purely for comments and has no handles. */}
        </Paper>
    );
});