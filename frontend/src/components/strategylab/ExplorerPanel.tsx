import React, { useState, useRef, forwardRef } from 'react';
import type { MouseEvent, FC } from 'react';
import { Box, Button, Collapse, Divider, List, ListItemButton, ListItemIcon, ListItemText, Typography, Menu, MenuItem } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import NoteAddIcon from '@mui/icons-material/NoteAdd';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import { Folder as FolderIcon, FolderOpen, File, Trash, DownloadIcon, Archive } from 'lucide-react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import JSZip from 'jszip';

// --- Data structure remains the same ---
export interface FileSystemItem {
  id: string;
  name: string;
  type: 'folder' | 'file';
  children?: FileSystemItem[];
  content?: string;
}

interface ImportedFile {
  name: string;
  content: string;
}

// --- Context Menu State now includes a 'type' for what was clicked ---
interface ContextMenuState {
  mouseX: number;
  mouseY: number;
  type: 'item' | 'container';
  item?: FileSystemItem;
}

// --- Recursive Component remains the same ---
const FileSystemTreeItem = forwardRef<HTMLDivElement, {
    item: FileSystemItem;
    onContextMenu: (event: MouseEvent, item: FileSystemItem) => void;
    onFileSelect: (fileId: string) => void;
    selectedFileId: string | null;
}>(({ item, onContextMenu, onFileSelect, selectedFileId }, ref ) => {

    const [open, setOpen] = useState(true);

    const handleItemClick = () => {
        if (item.type === 'folder') {
            setOpen(!open);
        } else {
            onFileSelect(item.id);
        }
    };

    const isSelected = item.type === 'file' && item.id === selectedFileId;

    const { attributes, listeners, setNodeRef: setDraggableRef, transform } = useDraggable({
        id: item.id,
        data: { 
            type: item.type,
        }
    });

    const { isOver, setNodeRef: setDroppableRef } = useDroppable({
        id: item.id,
        data: {
            type: 'folder'
        },
        disabled: item.type !== 'folder',
    });
    
    const style = transform ? {
        transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`,
        zIndex: 999, // Ensure the dragged item is on top
    } : undefined;

    return (
        <div ref={ref} style={style}>
            <ListItemButton
                ref={(node) => {
                    setDraggableRef(node);
                    if (item.type === 'folder') setDroppableRef(node);
                }}
                {...listeners}
                {...attributes}
                onClick={handleItemClick}
                onContextMenu={(e) => onContextMenu(e, item)}
                sx={{ 
                  pl: 1, 
                  height:30,
                  backgroundColor: isOver ? 'action-focus' : 'transparent',
                  transition: 'background-color 0.2s ease-in-out',
                }}
                selected={isSelected}
              >
                <ListItemIcon sx={{ minWidth: '24px', color: 'text.secondary' }}>
                    {item.type === 'folder' ? (open ? <FolderOpen size={16} /> : <FolderIcon size={16} />) : (<File size={16} />)}
                </ListItemIcon>
                
                <ListItemText primary={item.name} sx={{textWrap:'nowrap'}} primaryTypographyProps={{ sx: { fontSize: '0.8rem'} }}/>

            </ListItemButton>

            {item.type === 'folder' && (
                <Collapse in={open} timeout="auto" unmountOnExit>
                    <List component="div" disablePadding sx={{ pl: 2 }}>
                        {item.children?.map((child: FileSystemItem) => (
                            <FileSystemTreeItem
                                key={child.id}
                                item={child}
                                onContextMenu={onContextMenu}
                                onFileSelect={onFileSelect}
                                selectedFileId={selectedFileId}
                            />
                        ))}
                    </List>
                </Collapse>
            )}
        </div>
    );
});

export const ExplorerPanel: React.FC<{
      fileSystem: FileSystemItem[];
      onFileSelect: (fileId: string) => void;
      selectedFileId: string | null;
      onNewFile: (folderId?: string) => void;
      onNewFolder: (folderId?: string) => void;
      onDelete: (itemId: string) => void;
      onRename: (itemId: string, currentName: string) => void;
      onImportFiles: (files: ImportedFile[]) => void;
      onClearAll: () => void;
  }> = ({ fileSystem, onFileSelect, selectedFileId, onNewFile, onNewFolder, onDelete, onRename, onImportFiles, onClearAll }) => {
  
  const { setNodeRef: setRootDroppableRef } = useDroppable({
      id: 'root-droppable-area',
      data: {
          type: 'root-droppable',
      }
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportClick = () => {
    fileInputRef.current?.click();
    handleClose();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      return;
    }

    // FileReader is async, so we need to wrap it in a Promise
    // to handle multiple files correctly.
    const readAsText = (file: File): Promise<ImportedFile> => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          resolve({ name: file.name, content: reader.result as string });
        };
        reader.onerror = reject;
        reader.readAsText(file);
      });
    };

    // Create an array of promises, one for each file
    const promises = Array.from(files).map(readAsText);

    // Promise.all waits for all files to be read
    Promise.all(promises)
      .then(importedFiles => {
        // Then calls the parent handler with the complete array of results
        onImportFiles(importedFiles);
      })
      .catch(error => {
        console.error("Error reading one or more files:", error);
        alert("Failed to read one or more of the selected files.");
      });

    // Reset the input so the onChange event fires again for the same file(s)
    event.target.value = '';
  };

  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  
  const handleContainerContextMenu = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) {
        return;
    }
    event.preventDefault();
    setContextMenu({
        mouseX: event.clientX - 2,
        mouseY: event.clientY - 4,
        type: 'container', 
    });
  };

  // --- UPDATED: Context menu handler for specific items ---
  const handleItemContextMenu = (event: MouseEvent, item: FileSystemItem) => {
    event.preventDefault();
    event.stopPropagation(); // Stop the event from bubbling up to the container
    setContextMenu({
        mouseX: event.clientX - 2,
        mouseY: event.clientY - 4,
        type: 'item', // Set the type to 'item'
        item: item,
    });
  };

  const handleClose = () => {
    setContextMenu(null);
  };

  const handleNewFile = () => {
    // If the menu was opened on a folder, pass its ID as the parent. Otherwise, it's root level (undefined).
    const parentId = (contextMenu?.type === 'item' && contextMenu.item?.type === 'folder') 
      ? contextMenu.item.id 
      : undefined;
    onNewFile(parentId);
    handleClose();
  };
  
  const handleNewFolder = () => {
    const parentId = (contextMenu?.type === 'item' && contextMenu.item?.type === 'folder') 
      ? contextMenu.item.id 
      : undefined;
    onNewFolder(parentId);
    handleClose();
  };

  const handleDelete = () => {
      if(contextMenu?.item) {
          onDelete(contextMenu.item.id);
      }
      handleClose();
  };

  const handleRename = () => {
      if(contextMenu?.item) {
          onRename(contextMenu.item.id, contextMenu.item.name);
      }
      handleClose();
  };

  const handleExport = () => {
    if (!contextMenu?.item) return;

    const itemToExport = contextMenu.item;
    
    if (itemToExport.type === 'file') {
        // --- Handle single file export ---
        if (typeof itemToExport.content !== 'string') {
            alert("File has no content to export.");
            return;
        }
        
        // Create a blob from the file content
        const blob = new Blob([itemToExport.content], { type: 'text/plain;charset=utf-8' });
        
        // Create a temporary link element to trigger the download
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = itemToExport.name; // Set the download filename
        
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up
        URL.revokeObjectURL(link.href); // Free up memory

    } else if (itemToExport.type === 'folder') {
        // --- Handle folder export using jszip ---
        const zip = new JSZip();
        
        // A recursive function to add files and folders to the zip
        const addFolderToZip = (folder: FileSystemItem, currentZipFolder: JSZip) => {
            folder.children?.forEach(child => {
                if (child.type === 'file' && typeof child.content === 'string') {
                    currentZipFolder.file(child.name, child.content);
                } else if (child.type === 'folder') {
                    const newFolder = currentZipFolder.folder(child.name);
                    if (newFolder) {
                        addFolderToZip(child, newFolder);
                    }
                }
            });
        };
        
        // Start the zipping process from the selected folder
        addFolderToZip(itemToExport, zip);

        // Generate the zip file blob and trigger the download
        zip.generateAsync({ type: 'blob' }).then(content => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(content);
            link.download = `${itemToExport.name}.zip`;
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
        });
    }

    handleClose(); // Close the context menu
  };

  const handleExportAll = () => {
    const zip = new JSZip();

    // The recursive helper function is the same as before
    const addFolderToZip = (folder: FileSystemItem, currentZipFolder: JSZip) => {
        folder.children?.forEach(child => {
            if (child.type === 'file' && typeof child.content === 'string') {
                currentZipFolder.file(child.name, child.content);
            } else if (child.type === 'folder') {
                const newFolder = currentZipFolder.folder(child.name);
                if (newFolder) {
                    addFolderToZip(child, newFolder);
                }
            }
        });
    };

    // We iterate over the top-level `fileSystem` array
    fileSystem.forEach(item => {
        if (item.type === 'file' && typeof item.content === 'string') {
            zip.file(item.name, item.content);
        } else if (item.type === 'folder') {
            const newFolder = zip.folder(item.name);
            if (newFolder) {
                addFolderToZip(item, newFolder);
            }
        }
    });

    // Generate the zip file and trigger the download
    zip.generateAsync({ type: 'blob' }).then(content => {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        // Give it a generic name, e.g., based on the current date
        const date = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
        link.download = `dreamtester_strategies_${date}.zip`;
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
    });

    handleClose(); // Close the context menu
  };

  return (
    <Box sx={{ height: '100%', bgcolor: 'background.paper', p: 2, display: 'flex', flexDirection: 'column' }}>

      <Button variant="contained" color="primary" startIcon={<UploadFileIcon />} sx={{ mb: 2 }} onClick={handleImportClick}>
        Import Strategies
      </Button>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        accept=".py"
        multiple
      />

      <Button variant="outlined" color="primary" startIcon={<Archive />} sx={{ mb: 2 }} onClick={handleExportAll}>
        Export Strategies
      </Button>

      <Divider/>
      
      <Typography variant="h2" sx={{ mt: 2, mb: 1, px: 1 }}>Strategy Explorer</Typography>
      
      <Box 
        ref={setRootDroppableRef}
        onContextMenu={handleContainerContextMenu} 
        sx={{ flexGrow: 1 }}
      >
        <List component="nav" dense>
            {fileSystem.map(item => (
               <FileSystemTreeItem
                    key={item.id}
                    item={item}
                    onContextMenu={handleItemContextMenu}
                    onFileSelect={onFileSelect}
                    selectedFileId={selectedFileId}
                />
              ))}
        </List>
      </Box>

      <Menu
        open={contextMenu !== null}
        onClose={handleClose}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        {contextMenu?.type === 'container' && [
            <MenuItem key="new-file" onClick={handleNewFile}>
                <ListItemIcon><NoteAddIcon fontSize="small"/></ListItemIcon>
                <ListItemText>New File</ListItemText>
            </MenuItem>,
            <MenuItem key="new-folder" onClick={handleNewFolder}>
                <ListItemIcon><CreateNewFolderIcon fontSize="small"/></ListItemIcon>
                <ListItemText>New Folder</ListItemText>
            </MenuItem>,
            <MenuItem key="import" onClick={handleImportClick}>
                <ListItemIcon><UploadFileIcon fontSize="small"/></ListItemIcon>
                <ListItemText>Import Strategies</ListItemText>
            </MenuItem>,
            <MenuItem key="export-all" onClick={handleExportAll}>
                <ListItemIcon><Archive fontSize="small"/></ListItemIcon>
                <ListItemText>Export Strategies</ListItemText>
            </MenuItem>,
            <MenuItem key="clear-all" onClick={() => { onClearAll(); handleClose(); }} sx={{ color: 'error.main' }}>
                <ListItemIcon  sx={{color: 'error.main'}}><Trash fontSize="small"/></ListItemIcon>
                <ListItemText>Clear All</ListItemText>
            </MenuItem>
        ]}
        {contextMenu?.type === 'item' && contextMenu.item?.type === 'folder' && [
            <MenuItem key="new-file" onClick={handleNewFile}>New File</MenuItem>,
            <MenuItem key="new-folder" onClick={handleNewFolder}>New Folder</MenuItem>,
            <Divider key="divider" />,
        ]}
        {contextMenu?.type === 'item' && [
             <MenuItem key="rename" onClick={handleRename}>Rename</MenuItem>,
             <MenuItem key="export" onClick={handleExport}>
                <ListItemText>Export</ListItemText>
             </MenuItem>,
             <MenuItem key="delete" onClick={handleDelete} sx={{ color: 'error.main' }}>Delete</MenuItem>
        ]}
      </Menu>
    </Box>
  );
};