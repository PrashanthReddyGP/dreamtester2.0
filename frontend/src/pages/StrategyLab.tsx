import React, { useState, useCallback, useEffect, useMemo } from 'react';
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

import {  submitBatchBacktest } from '../services/api';
import type { StrategyFilePayload, BatchSubmitPayload, BatchSubmitResponse } from '../services/api';
import { useNavigate } from 'react-router-dom';
import { useTerminal } from '../context/TerminalContext';
import { useAnalysis } from '../context/AnalysisContext';

import { OptimizeModal } from '../components/strategylab/OptimizeModal';
import type { OptimizationConfig, TestSubmissionConfig, SuperOptimizationConfig } from '../components/strategylab/OptimizeModal';

import { DurabilityModal } from '../components/strategylab/DurabilityModal';
import type { DurabilitySubmissionConfig } from '../components/strategylab/DurabilityModal';

import { HedgeModal } from '../components/strategylab/HedgeModal';
import type { HedgeOptimizationConfig } from '../components/strategylab/HedgeModal';

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

const findFirstFile = (nodes: FileSystemItem[]): FileSystemItem | null => {
    for (const node of nodes) {
        if (node.type === 'file') {
            return node; // Found the first file, return it
        }
        if (node.children) {
            const foundInChild = findFirstFile(node.children);
            if (foundInChild) {
                return foundInChild; // A file was found in a subfolder
            }
        }
    }
    return null; // No file found in this branch
};

// Helper to flatten the file tree (can be inside or outside the component)
const getAllFiles = (nodes: FileSystemItem[]): FileSystemItem[] => {
    let files: FileSystemItem[] = [];
    for (const node of nodes) {
        if (node.type === 'file') {
            files.push(node);
        }
        if (node.children) {
            files = files.concat(getAllFiles(node.children));
        }
    }
    return files;
};

export const StrategyLab: React.FC = () => {
    const { connectToBatch, toggleTerminal } = useTerminal();
    const { clearResults } = useAnalysis();

    const [fileSystem, setFileSystem] = useState<FileSystemItem[]>([]);
    const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [currentEditorCode, setCurrentEditorCode] = useState<string>('// Select a file to begin...');
    const [isClearConfirmOpen, setIsClearConfirmOpen] = useState(false);
    const [isBacktestRunning, setIsBacktestRunning] = useState(false);
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [isOptimizeModalOpen, setIsOptimizeModalOpen] = useState(false);
    const [isDurabilityModalOpen, setIsDurabilityModalOpen] = useState(false);

    const [isHedgeModalOpen, setIsHedgeModalOpen] = useState(false);

    const navigate = useNavigate();

    const handleOpenClearConfirm = () => setIsClearConfirmOpen(true);
    const handleCloseClearConfirm = () => setIsClearConfirmOpen(false);

    const [localCsvFile, setLocalCsvFile] = useState<File | null>(null);
    const [localCsvData, setLocalCsvData] = useState<string | null>(null);

    const handleClearCsv = () => {
        setLocalCsvFile(null);
        setLocalCsvData(null);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setLocalCsvFile(file);
            const reader = new FileReader();
            reader.onload = (e) => {
                const text = e.target?.result;
                setLocalCsvData(text as string);
            };
            reader.readAsText(file);
        }
        // Reset the input value to allow selecting the same file again
        event.target.value = '';
    };

    const handleRunBacktestWithCsv = async () => {
        if (!selectedFileId || !localCsvData) {
            alert("Please select a strategy file and a CSV data file.");
            return;
        }
        
        setIsBacktestRunning(true);
        toggleTerminal(true);

        try {
            clearResults();
            await handleSaveContent(); // Save any pending changes first

            const payload = {
                strategy_code: currentEditorCode,
                strategy_name: findFileById(fileSystem, selectedFileId)?.name || 'local_run',
                csv_data: localCsvData,
            };

            // This will be a new API call
            const response = await fetch(`${API_URL}/api/backtest/local-submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to submit local backtest.');
            }
            
            const result = await response.json();
            if (result.batch_id) {
                connectToBatch(result.batch_id);
                navigate('/analysis');
            }
        } catch (error) {
            console.error("Failed to run backtest with CSV:", error);
            alert(`Error: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
        } finally {
            setIsBacktestRunning(false);
        }
    };

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

    const loadStrategies = useCallback(async (): Promise<FileSystemItem[]> => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/strategies`);
            if (!response.ok) throw new Error('Failed to fetch strategies');
            const data = await response.json();
            setFileSystem(data);
            return data;
        } catch (error) {
            console.error("Error loading strategies:", error);
            return [];
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
    
    useEffect(() => {
        // Condition 1: Data has just finished loading
        // Condition 2: The file system is not empty
        // Condition 3: No file is currently selected
        if (!isLoading && fileSystem.length > 0 && !selectedFileId) {
            // Find the very first file in the entire tree structure
            const firstFile = findFirstFile(fileSystem);
            if (firstFile) {
                // Set it as the selected file
                setSelectedFileId(firstFile.id);
            }
        }
    // This effect should run whenever isLoading or fileSystem changes.
    }, [isLoading, fileSystem, selectedFileId]);

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

    const handleSaveContent = async (): Promise<FileSystemItem[]> => {
        if (!selectedFileId) {
            console.warn("handleSaveContent called with no file selected.");
            return fileSystem; // Return the current state if there's nothing to save
        }
        try {
            await fetch(`${API_URL}/api/strategies/${selectedFileId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                // Use the code from our state variable
                body: JSON.stringify({ content: currentEditorCode }), 
            });
            const freshFileSystem = await loadStrategies(); // Re-fetch to confirm save
            console.log("Strategy saved successfully!");
            return freshFileSystem;
        } catch (error) {
            console.error("Failed to save content:", error);
            alert("Error: Could not save strategy.");
            throw error;
        }
    };

    const hasUnsavedChanges = (): boolean => {
        if (!selectedFileId) return false;
        const savedFile = findFileById(fileSystem, selectedFileId);
        // Compare the code in the editor with the code from the last loaded fileSystem state
        return savedFile?.content !== currentEditorCode;
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

    const handleRunBacktest = async (useTrainingSet: boolean) => {

        if (!selectedFileId) {
            alert("Please select a strategy file before running a backtest.");
            return;
        }

        setIsBacktestRunning(true);
        toggleTerminal(true);

        try {
            clearResults();

            let updatedFileSystem = fileSystem;

            if (hasUnsavedChanges()) {
                console.log("Unsaved changes detected. Saving open file before batch backtest...");
                updatedFileSystem = await handleSaveContent();
            }

            // Filter the top-level fileSystem array directly.
            // We only want items where the type is 'file'.
            const rootFiles: StrategyFilePayload[] = updatedFileSystem
                .filter(item => item.type === 'file' && item.content)
                .map(file => ({
                    id: file.id,
                    name: file.name,
                    content: file.content!,
                }));

            if (rootFiles.length === 0) {
                alert("No strategies found to backtest.");
                setIsBacktestRunning(false); // Reset loading state
                return;
            }

            const payload: BatchSubmitPayload = {
                strategies: rootFiles,
                use_training_set: useTrainingSet,
            };
            
            // Call the API service with the filtered list of root files
            const result = await submitBatchBacktest(payload);
            console.log("Batch backtest submitted successfully!", result);

            if (result.batch_id) {
                connectToBatch(result.batch_id);
            } else {
                // This case should ideally not happen if the API call was successful
                console.error("No batch_id received from server!");
            }

            navigate('/analysis');

        } catch (error) {
            console.error("Failed to run batch backtest:", error);
            alert(`Error: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
        } finally {
            setIsBacktestRunning(false);
        }
    };

    const handleBacktestSingleFile = async (fileToRun: FileSystemItem) => {
        if (!fileToRun || fileToRun.type !== 'file') {
            alert("Invalid file provided for backtest.");
            return;
        }

        console.log(`Running single backtest for: ${fileToRun.name}`);
        
        setIsBacktestRunning(true);
        toggleTerminal(true);

        try {
            clearResults();

            let finalFileContent = fileToRun.content;
            
            // 1. Check for unsaved changes
            if (hasUnsavedChanges()) {
                console.log("Unsaved changes detected. Saving open file before single backtest...");
                const updatedFileSystem = await handleSaveContent();
                
                // If the file we want to run is the one that was just saved,
                // we need to get its updated content from the fresh file system.
                if (fileToRun.id === selectedFileId) {
                    const freshlySavedFile = findFileById(updatedFileSystem, fileToRun.id);
                    finalFileContent = freshlySavedFile?.content;
                }
            }

            if (!finalFileContent) {
                alert("File has no content to backtest.");
                setIsBacktestRunning(false);
                return;
            }

            const fileToBacktest: StrategyFilePayload = {
                id: fileToRun.id,
                name: fileToRun.name,
                content: finalFileContent,
            };

            // The `submitBatchBacktest` API can handle an array with just one item
            const payload: BatchSubmitPayload = {
                strategies: [fileToBacktest],
                use_training_set: true, // Defaulting to true for context-menu runs
            };

            const result = await submitBatchBacktest(payload);
            
            console.log("Single file backtest submitted successfully!", result);

            if (result.batch_id) {
                connectToBatch(result.batch_id);
            } else {
                console.error("No batch_id received from server!");
            }

            navigate('/analysis');

        } catch (error) {
            console.error("Failed to run single file backtest:", error);
            alert(`Error: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
        } finally {
            setIsBacktestRunning(false);
        }
    };
    
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

    const handleOpenOptimizeModal = () => {
        if (!selectedFileId) {
            alert("Please select a strategy file to optimize.");
            return;
        }
        setIsOptimizeModalOpen(true);
    };

    const handlOpenDurabilityModal = () => {
        if (!selectedFileId) {
            alert("Please select a strategy file to optimize.");
            return;
        }
        setIsDurabilityModalOpen(true);
    };

    const handleCloseOptimizeModal = () => {
        setIsOptimizeModalOpen(false);
    };
    
    const handleCloseDurabilityModal = () => {
        setIsDurabilityModalOpen(false);
    };

    // const handleRunOptimization = async (config: OptimizationConfig) => {
    //     // 1. Set initial loading states and provide feedback
    //     console.log("Submitting optimization with config:", config);
    //     setIsOptimizing(true);

    //     try {
    //         // 2. Prepare the application state for a new run
    //         clearResults(); // Clear any previous results in the Analysis Hub

    //         await handleSaveContent(); // Best practice: ensure the latest code is saved on the server

    //         // 3. Make the actual API call to the backend
    //         const response = await fetch(`${API_URL}/api/optimize/submit`, {
    //             method: 'POST',
    //             headers: {
    //                 'Content-Type': 'application/json',
    //             },
    //             body: JSON.stringify(config),
    //         });

    //         // 4. Handle non-successful HTTP responses (e.g., 400, 500 errors)
    //         if (!response.ok) {
    //             // Try to get a detailed error message from the backend's JSON response
    //             const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
    //             // Throw an error to be caught by the catch block
    //             throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
    //         }

    //         // 5. Handle the successful response
    //         const result = await response.json();

    //         // 6. Connect to the WebSocket stream and navigate to the results page
    //         if (result.batch_id) {
    //             connectToBatch(result.batch_id); // This function is from your useTerminal context
    //             navigate('/analysis'); // Redirect the user to see the results stream in
    //             toggleTerminal(true); // Open the terminal to show live logs from the backend
    //         } else {
    //             // This is an important edge case to handle
    //             throw new Error("Submission was successful, but no batch ID was returned from the server.");
    //         }

    //     } catch (error) {
    //         // 7. Catch any errors (from the network, API, or thrown manually) and display them
    //         console.error("Failed to run optimization:", error);
    //         // Use a user-friendly alert to show the error
    //         alert(`Error submitting optimization: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
    //     } finally {
    //         // 8. This block ALWAYS runs, ensuring the UI is cleaned up correctly
    //         setIsOptimizing(false);       // Reset the loading state on the button
    //         handleCloseOptimizeModal(); // Close the modal, whether the submission succeeded or failed
    //     }
    // };

    const handleRunOptimization = async (config: SuperOptimizationConfig) => {
        setIsOptimizing(true);
        console.log("Submitting optimization with config:", config);

        try {
            clearResults();
            await handleSaveContent();

            let response;
            
            // Call the parameter optimization endpoint
            response = await fetch(`${API_URL}/api/optimize/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            
            if (!response.ok) {
                // Try to get a detailed error message from the backend's JSON response
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                // Throw an error to be caught by the catch block
                throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
            }

            // 5. Handle the successful response
            const result = await response.json();

            // 6. Connect to the WebSocket stream and navigate to the results page
            if (result.batch_id) {
                connectToBatch(result.batch_id); // This function is from your useTerminal context
                navigate('/analysis'); // Redirect the user to see the results stream in
                toggleTerminal(true); // Open the terminal to show live logs from the backend
            } else {
                // This is an important edge case to handle
                throw new Error("Submission was successful, but no batch ID was returned from the server.");
            }

        } catch (error) {
            console.error("Failed to run optimization:", error);
            // Use a user-friendly alert to show the error
            alert(`Error submitting optimization: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
        } finally {
            setIsOptimizing(false);
            handleCloseOptimizeModal();
        }
    };

    const handleRunDurabilityTest = async (config: DurabilitySubmissionConfig) => {
        setIsOptimizing(true);
        console.log("Submitting Durability Test with config:", config);

        try {
            clearResults();
            await handleSaveContent();

            // IMPORTANT: This will likely be a new endpoint on your backend
            // to handle these more complex test types.
            const response = await fetch(`${API_URL}/api/durability/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
            }

            const result = await response.json();

            if (result.batch_id) {
                connectToBatch(result.batch_id);
                navigate('/analysis');
                toggleTerminal(true);
            } else {
                throw new Error("Submission was successful, but no batch ID was returned from the server.");
            }

        } catch (error) {
            console.error("Failed to run durability test:", error);
            alert(`Error submitting test: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
        } finally {
            setIsOptimizing(false);
            // Make sure to close the correct modal
            handleCloseDurabilityModal();
        }
    };

    // --- Add handlers for the new modal ---
    const handleOpenHedgeModal = () => {
        if (!selectedFileId) {
            alert("Please select a primary strategy (Strategy A) from the explorer first.");
            return;
        }
        setIsHedgeModalOpen(true);
    };
    
    const handleCloseHedgeModal = () => setIsHedgeModalOpen(false);

    // --- Add the new submission handler for hedge optimizations ---
    const handleRunHedgeOptimization = async (config: HedgeOptimizationConfig) => {
        setIsOptimizing(true);
        toggleTerminal(true);
        console.log("Submitting Hedge Optimization with config:", config);
        
        try {
            clearResults();
            await handleSaveContent(); // Save any open files

            const response = await fetch(`${API_URL}/api/optimize/hedge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to submit hedge optimization.');
            }

            const result = await response.json();
            if (result.batch_id) {
                connectToBatch(result.batch_id);
                navigate('/analysis');
            } else {
                throw new Error("Submission successful, but no batch ID was returned.");
            }

        } catch (error) {
            console.error("Failed to run hedge optimization:", error);
            alert(`Error: ${error instanceof Error ? error.message : 'An unknown error occurred.'}`);
        } finally {
            setIsOptimizing(false);
            handleCloseHedgeModal();
        }
    };

    // --- Prepare data for the HedgeModal props ---
    const selectedFile = useMemo(() => {
        return selectedFileId ? findFileById(fileSystem, selectedFileId) : null;
    }, [selectedFileId, fileSystem]);

    const allAvailableFiles = useMemo(() => getAllFiles(fileSystem), [fileSystem]);

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <Box sx={{ height: '100%', width: '100vw' }}>
            <PanelGroup direction="horizontal">
                <Panel defaultSize={15} minSize={15}>
                    <ExplorerPanel
                        fileSystem={fileSystem}
                        onFileSelect={handleFileSelect}
                        selectedFileId={selectedFileId}
                        onBacktestFile={handleBacktestSingleFile}
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
                <Panel defaultSize={16} minSize={16}>
                    <SettingsPanel 
                        onSave={handleSaveContent}
                        isSaveDisabled={!selectedFileId}
                        onRunBacktest={handleRunBacktest}
                        onRunBacktestWithCsv={handleRunBacktestWithCsv}
                        onOptimizeStrategy={handleOpenOptimizeModal}
                        onDurabilityTests={handlOpenDurabilityModal}
                        onHedgeOptimize={handleOpenHedgeModal}
                        isBacktestRunning={isBacktestRunning}
                        onFileChange={handleFileChange}
                        onClearCsv={handleClearCsv}
                        selectedCsvFile={localCsvFile}
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

            <OptimizeModal
                open={isOptimizeModalOpen}
                onClose={handleCloseOptimizeModal}
                onSubmit={handleRunOptimization}
                strategyCode={currentEditorCode}
                isSubmitting={isOptimizing}
            />

            <DurabilityModal
                open={isDurabilityModalOpen}
                onClose={handleCloseDurabilityModal}
                onSubmit={handleRunDurabilityTest}
                strategyCode={currentEditorCode}
                isSubmitting={isOptimizing}
            />

            <HedgeModal
                open={isHedgeModalOpen}
                onClose={handleCloseHedgeModal}
                onSubmit={handleRunHedgeOptimization}
                isSubmitting={isOptimizing}
                initialStrategy={selectedFile}
                availableStrategies={allAvailableFiles}
            />

        </Box>
    </DndContext>
    );
};
