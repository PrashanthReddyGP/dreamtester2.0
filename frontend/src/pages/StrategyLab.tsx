import React, { useState, useCallback, useEffect } from 'react';
import { Box } from '@mui/material';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import { v4 as uuidv4 } from 'uuid';

import { ExplorerPanel } from '../components/strategylab/ExplorerPanel';
import type { FileSystemItem } from '../components/strategylab/ExplorerPanel';
import { EditorPanel } from '../components/strategylab/EditorPanel';
import { SettingsPanel } from '../components/strategylab/SettingsPanel';
import { NameInputDialog  } from '../components/strategylab/NameItemDialog';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import { ConfirmationDialog } from '../components/common/ConfirmationDialog';

const API_URL = 'http://127.0.0.1:8000';

interface ImportedFile {
  name: string;
  content: string;
}


// A styled resize handle that is more visible on hover for better UX
const ResizeHandle = () => {
  return (
    <PanelResizeHandle
      style={{
        width: '4px',
        background: 'transparent',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
        <Box sx={{
          width: '4px',
          height: '40px',
          borderRadius: '2px',
          backgroundColor: 'action.hover',
          transition: 'background-color 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: 'divider'
          }
        }} />
    </PanelResizeHandle>
  );
};

// --- 1. LIFTED STATE: Define the initial file system with content here ---
const rsiMomentumCode = `print("This is the RSI Momentum strategy template code.")`;

const initialFileSystem: FileSystemItem[] = [
  {
    id: uuidv4(),
    name: 'Templates',
    type: 'folder',
    children: [
      { id: 'rsi-momentum-id', name: 'RSI_Momentum.py', type: 'file', content: rsiMomentumCode },
    ],
  }
];

// Helper function to find a file by ID in the tree
const findFileById = (nodes: FileSystemItem[], id: string): FileSystemItem | null => {
    for (const node of nodes) {
        if (node.type === 'file' && node.id === id) return node;
        if (node.children) {
            const found = findFileById(node.children, id);
            if (found) return found;
        }
    }
    return null;
}

export const StrategyLab: React.FC = () => {
    const [fileSystem, setFileSystem] = useState<FileSystemItem[]>([]);
    const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
    const [editorCode, setEditorCode] = useState<string>('// Select a file to begin...');
    const [isLoading, setIsLoading] = useState(true);
    const [currentEditorCode, setCurrentEditorCode] = useState<string>('// Select a file to begin...');
    const [isClearConfirmOpen, setIsClearConfirmOpen] = useState(false);

    const handleOpenClearConfirm = () => setIsClearConfirmOpen(true);
    const handleCloseClearConfirm = () => setIsClearConfirmOpen(false);

    const [deleteConfirmState, setDeleteConfirmState] = useState<{ open: boolean; itemId: string | null }>({
        open: false,
        itemId: null,
    });

    const handleOpenDeleteConfirm = (itemId: string) => {
        setDeleteConfirmState({ open: true, itemId });
    };

    // This function closes the dialog
    const handleCloseDeleteConfirm = () => {
        setDeleteConfirmState({ open: false, itemId: null });
    };

    const [dialogState, setDialogState] = useState({
      open: false,
      type: 'file' as 'file' | 'folder',
      parentId: undefined as string | undefined,
    });

    const [createDialogState, setCreateDialogState] = useState({
      open: false,
      type: 'file' as 'file' | 'folder',
      parentId: undefined as string | undefined,
    });

    const [renameDialogState, setRenameDialogState] = useState({
      open: false,
      itemId: '',
      currentName: '',
    });

    const handleOpenNewItemDialog = (type: 'file' | 'folder', parentId?: string) => {
        setCreateDialogState({ open: true, type, parentId });
    };

    const handleCloseNewItemDialog = () => {
        setCreateDialogState({ ...createDialogState, open: false });
    };
    
    const handleFileSelect = (fileId: string) => {
        setSelectedFileId(fileId);
    };

    const loadStrategies = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/strategies`);
            if (!response.ok) throw new Error('Failed to fetch strategies');
            const data = await response.json();
            setFileSystem(data);
        } catch (error) {
            console.error("Error loading strategies:", error);
            // Handle error display if needed
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadStrategies();
    }, [loadStrategies]);

    useEffect(() => {
        const selectedFile = selectedFileId ? findFileById(fileSystem, selectedFileId) : null;
        setCurrentEditorCode(selectedFile?.content ?? '// Select a file from the explorer to view its content.');
    }, [selectedFileId, fileSystem]);

    const handleCreateItem = async (name: string) => {
        const newItem = {
            id: uuidv4(),
            name,
            type: createDialogState.type,
            parentId: createDialogState.parentId,
        };

        try {
            await fetch(`${API_URL}/api/strategies`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newItem),
            });
            await loadStrategies();
        } catch (error) {
            console.error(`Failed to create ${dialogState.type}:`, error);
        }
    };
    
    const handleNewItem = async (type: 'file' | 'folder', parentId?: string) => {
        const name = prompt(`Enter name for new ${type}:`, type === 'file' ? 'new_strategy.py' : 'New_Folder');
        if (!name) return;

        const newItem = {
            id: uuidv4(),
            name,
            type,
            parentId: parentId,
        };

        try {
            await fetch(`${API_URL}/api/strategies`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newItem),
            });
            await loadStrategies(); // Re-fetch the tree to show the new item
        } catch (error) {
            console.error(`Failed to create ${type}:`, error);
        }
    };

    const handleConfirmDelete = async () => {
        const { itemId } = deleteConfirmState;
        if (!itemId) return; // Safety check

        try {
            await fetch(`${API_URL}/api/strategies/${itemId}`, { method: 'DELETE' });
            if (selectedFileId === itemId) {
                setSelectedFileId(null); // Deselect if the deleted file was open
            }
            await loadStrategies(); // Re-fetch to update UI
        } catch (error) {
            console.error("Failed to delete item:", error);
        }
    };

    const handleRename = async (itemId: string, oldName: string) => {
        const newName = prompt("Enter new name:", oldName);
        if (!newName || newName === oldName) return;
        try {
            await fetch(`${API_URL}/api/strategies/${itemId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName }),
            });
            await loadStrategies();
        } catch (error) {
            console.error("Failed to rename item:", error);
        }
    };

    const handleSaveContent = async () => {
        if (!selectedFileId) {
            alert("No file selected to save.");
            return;
        }
        try {
            await fetch(`${API_URL}/api/strategies/${selectedFileId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                // Use the code from our state variable
                body: JSON.stringify({ content: currentEditorCode }), 
            });
            await loadStrategies(); // Re-fetch to confirm save
            console.log("Strategy saved successfully!");
        } catch (error) {
            console.error("Failed to save content:", error);
            alert("Error: Could not save strategy.");
        }
    };

    const handleOpenRenameDialog = (itemId: string, currentName: string) => {
        setRenameDialogState({ open: true, itemId, currentName });
    };

    const handleCloseRenameDialog = () => {
        setRenameDialogState({ ...renameDialogState, open: false, itemId: '', currentName: '' });
    };

    const handleRenameItem = async (newName: string) => {
        const { itemId, currentName } = renameDialogState;
        if (!newName || newName === currentName) {
            return; // No need to make an API call if the name hasn't changed
        }

        try {
            await fetch(`${API_URL}/api/strategies/${itemId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName }),
            });
            await loadStrategies();
        } catch (error) {
            console.error("Failed to rename item:", error);
        }
    };

    const handleImportFile = async (name: string, content: string) => {
        // We'll create the new file at the root level for simplicity
        const newItem = {
            id: uuidv4(),
            name,
            type: 'file' as 'file',
            content, 
            parentId: undefined,
        };

        try {
            await fetch(`${API_URL}/api/strategies`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newItem),
            });
            await loadStrategies(); // Refresh the explorer to show the new file
        } catch (error) {
            console.error(`Failed to import file:`, error);
            alert("Error: Could not import the strategy file.");
        }
    };

    const handleImportFiles = async (files: ImportedFile[]) => {
        // Map the imported file data to the format our API expects
        const newItemsToCreate = files.map(file => ({
            id: uuidv4(),
            name: file.name,
            type: 'file' as 'file',
            content: file.content,
            parentId: undefined, // Import to root level
        }));
        
        if (newItemsToCreate.length === 0) return;

        try {
            // --- CALL THE NEW BULK ENDPOINT ---
            await fetch(`${API_URL}/api/strategies/bulk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newItemsToCreate),
            });
            await loadStrategies(); // Refresh the explorer
        } catch (error) {
            console.error(`Failed to import files:`, error);
            alert("Error: Could not import the strategy files.");
        }
    };

    // --- 2. SETUP DND SENSORS (RECOMMENDED FOR POINTER DEVICES) ---
    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                // Require pointer to move 5px before activating a drag
                distance: 5,
            },
        })
    );

    // --- 3. CREATE THE API HANDLER FOR MOVING AN ITEM ---
    const handleMoveItem = async (itemId: string, newParentId: string | null) => {
        try {
            await fetch(`${API_URL}/api/strategies/${itemId}/move`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ newParentId: newParentId }),
            });
            // --- Crucially, re-fetch the truth from the sorted backend ---
            await loadStrategies();
        } catch (error) {
            console.error("Failed to move item:", error);
            alert("Error: Could not move the item.");
        }
    };

    // --- 4. CREATE THE MAIN DRAG HANDLER ---
    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;

        // If not dropped on a valid target, do nothing
        if (!over) return;

        // If dropped on itself, do nothing
        if (active.id === over.id) return;
        
        const activeId = active.id as string;
        const overId = over.id as string;

        // Find the file being dropped on
        let newParentId: string | null = null;
        if (over.data.current?.type === 'folder') {
            // Dropped onto a folder
            newParentId = overId;
        } else if (over.data.current?.type === 'root-droppable') {
            // Dropped onto the root container
            newParentId = null;
        } else {
            // Dropped onto a file, which is invalid. Do nothing.
            // A better UX might be to find the file's parent folder, but this is safer.
            console.log("Invalid drop target (cannot drop on a file).");
            return;
        }

        handleMoveItem(activeId, newParentId);
    };

    const handleConfirmClearAll = async () => {
        try {
            await fetch(`${API_URL}/api/strategies`, { method: 'DELETE' });
            // Refresh the UI to show it's empty
            await loadStrategies();
        } catch (error) {
            console.error("Failed to clear all strategies:", error);
            alert("Error: Could not clear strategies.");
        }
    };

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <Box sx={{ height: '100%', width: '100vw' }}>
            <PanelGroup direction="horizontal">
                <Panel defaultSize={15} minSize={15}>
                    <ExplorerPanel
                        fileSystem={fileSystem}
                        onFileSelect={handleFileSelect}
                        selectedFileId={selectedFileId}
                        onNewFile={(folderId) => handleOpenNewItemDialog('file', folderId)}
                        onNewFolder={(folderId) => handleOpenNewItemDialog('folder', folderId)}
                        onDelete={handleOpenDeleteConfirm}
                        onRename={handleOpenRenameDialog}
                        onImportFiles={handleImportFiles}
                        onClearAll={handleOpenClearConfirm}
                    />
                </Panel>
                <ResizeHandle />
                <Panel defaultSize={70} minSize={40}>
                    <EditorPanel
                        fileId={selectedFileId}
                        code={currentEditorCode}
                        onChange={setCurrentEditorCode}
                    />
                </Panel>
                <ResizeHandle />
                <Panel defaultSize={15} minSize={15}>
                  <SettingsPanel 
                    onSave={handleSaveContent}
                    isSaveDisabled={!selectedFileId} 
                  />
                </Panel>
            </PanelGroup>

            <ConfirmationDialog
                open={isClearConfirmOpen}
                onClose={handleCloseClearConfirm}
                onConfirm={handleConfirmClearAll}
                title="Clear All Strategies?"
                message="Are you sure you want to delete all files and folders? This action is permanent and cannot be undone."
            />

            <ConfirmationDialog
                open={deleteConfirmState.open}
                onClose={handleCloseDeleteConfirm}
                onConfirm={handleConfirmDelete}
                title="Delete Item?"
                message="Are you sure you want to delete this specific item? This action cannot be undone."
            />


            <NameInputDialog
                open={createDialogState.open}
                onClose={handleCloseNewItemDialog}
                onConfirm={handleCreateItem}
                dialogTitle={`Create New ${createDialogState.type}`}
                dialogText={`Please enter a name for the new ${createDialogState.type}.`}
                confirmButtonText="Create"
                initialValue={createDialogState.type === 'file' ? 'new_strategy.py' : 'New_Folder'}
            />
            <NameInputDialog
                open={renameDialogState.open}
                onClose={handleCloseRenameDialog}
                onConfirm={handleRenameItem}
                dialogTitle="Rename Item"
                dialogText="Please enter a new name for the item."
                confirmButtonText="Rename"
                initialValue={renameDialogState.currentName}
            />

        </Box>
    </DndContext>
    );
};
