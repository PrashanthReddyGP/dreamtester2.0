import React, { useState } from 'react';
import { Box, Button, Menu, MenuItem, Tooltip } from '@mui/material';
import { Save, SaveAs, FolderOpen, Delete } from '@mui/icons-material';
import { usePipeline } from '../../context/PipelineContext';
import { NameInputDialog } from '../common/NameItemDialog';
import NoteAddIcon from '@mui/icons-material/NoteAdd';

export const PipelineControls: React.FC = () => {
    const { savedWorkflows, saveWorkflow, loadWorkflow, clearBackendCache, newWorkflow, currentWorkflowName } = usePipeline();
    const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
    const [loadMenuAnchorEl, setLoadMenuAnchorEl] = useState<null | HTMLElement>(null);

    const handleSaveConfirm = async (name: string) => {
        await saveWorkflow(name);
        setIsSaveDialogOpen(false);
    };

    const handleLoadClick = (event: React.MouseEvent<HTMLElement>) => {
        setLoadMenuAnchorEl(event.currentTarget);
    };

    const handleLoadClose = () => {
        setLoadMenuAnchorEl(null);
    };

    const handleLoadSelect = (name: string) => {
        loadWorkflow(name);
        handleLoadClose();
    };

    // New handler for the "Save" button
    const handleSaveClick = async () => {
        // If there is a current workflow name, save directly (overwrite).
        if (currentWorkflowName) {
            await saveWorkflow(currentWorkflowName);
            // Optional: You could add a success notification (e.g., a snackbar) here.
        } else {
            // If it's a new workflow (no name), open the dialog to ask for a name.
            setIsSaveDialogOpen(true);
        }
    };
    return (
        <>
            <Box 
                sx={{ 
                    position: 'absolute', 
                    top: 16, 
                    left: 16, 
                    zIndex: 10, 
                    display: 'flex',
                    gap: 1,
                    borderRadius: 2,
                }}
            >
                <Tooltip title="New">
                    <Button 
                        variant="outlined" 
                        sx={{
                            color: '#e7e7e7ff',
                            backgroundColor: 'background.paper'
                        }}
                        onClick={newWorkflow}
                        startIcon={<NoteAddIcon />}
                        >
                        New
                    </Button>
                </Tooltip>
                
                <Tooltip title="Load Workflow">
                    <Button 
                        variant="outlined" 
                        sx={{
                            color: '#e7e7e7ff',
                            backgroundColor: 'background.paper'
                        }}
                        onClick={handleLoadClick} 
                        disabled={savedWorkflows.length === 0}
                        startIcon={<FolderOpen />}
                        >
                        Load
                    </Button>
                </Tooltip>

                <Tooltip title="Save Workflow">
                    <Button 
                        variant="outlined" 
                        sx={{
                            color: '#e7e7e7ff',
                            backgroundColor: 'background.paper'
                        }}
                        onClick={handleSaveClick}
                        startIcon={<Save />}
                    >
                        Save
                    </Button>
                </Tooltip>

                <Tooltip title="Save Workflow As">
                    <Button 
                        variant="outlined" 
                        sx={{
                            color: '#e7e7e7ff',
                            backgroundColor: 'background.paper'
                        }}
                        onClick={() => setIsSaveDialogOpen(true)}
                        startIcon={<SaveAs />}
                    >
                        Save As
                    </Button>
                </Tooltip>

                <Tooltip title="Clear Backend Cache">
                    <Button 
                        variant="outlined" 
                        sx={{
                            color: '#e7e7e7ff',
                            backgroundColor: 'background.paper'
                        }}
                        onClick={clearBackendCache}
                        startIcon={<Delete />}
                        >
                        Clear Cache
                    </Button>
                </Tooltip>
                
                <Menu
                    anchorEl={loadMenuAnchorEl}
                    open={Boolean(loadMenuAnchorEl)}
                    onClose={handleLoadClose}
                >
                    {savedWorkflows.length > 0 ? (
                        savedWorkflows.map(name => (
                            <MenuItem key={name} onClick={() => handleLoadSelect(name)}>
                                {name}
                            </MenuItem>
                        ))
                    ) : (
                        <MenuItem disabled>No saved workflows</MenuItem>
                    )}
                </Menu>
            </Box>

            <NameInputDialog
                open={isSaveDialogOpen}
                onClose={() => setIsSaveDialogOpen(false)}
                onConfirm={handleSaveConfirm}
                dialogTitle="Save Workflow As..."
                label="Workflow Name"
                confirmButtonText="Save"
            />
        </>
    );
};