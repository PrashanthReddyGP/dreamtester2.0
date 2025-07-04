import React, { useState } from 'react';
import type { MouseEvent, FC } from 'react';
import { Box, Button, Collapse, Divider, List, ListItemButton, ListItemIcon, ListItemText, Typography, Menu, MenuItem } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CodeIcon from '@mui/icons-material/Code';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import NoteAddIcon from '@mui/icons-material/NoteAdd';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import { v4 as uuidv4 } from 'uuid';

// --- Data structure remains the same ---
export interface FileSystemItem {
  id: string;
  name: string;
  type: 'folder' | 'file';
  children?: FileSystemItem[];
  content?: string; // Files can have content
}

// --- Context Menu State now includes a 'type' for what was clicked ---
interface ContextMenuState {
  mouseX: number;
  mouseY: number;
  type: 'item' | 'container'; // NEW: Differentiate the click source
  item?: FileSystemItem; // item is now optional
}

// --- Recursive Component remains the same ---
const FileSystemTreeItem: FC<{
    item: FileSystemItem;
    onContextMenu: (event: MouseEvent, item: FileSystemItem) => void;
    onFileSelect: (fileId: string) => void;
    selectedFileId: string | null;
}> = ({ item, onContextMenu, onFileSelect, selectedFileId }) => {
    const [open, setOpen] = useState(true);

    const handleItemClick = () => {
        if (item.type === 'folder') {
            setOpen(!open);
        } else {
            // This is a file, so we call the selection handler
            onFileSelect(item.id);
        }
    };
    const isSelected = item.type === 'file' && item.id === selectedFileId;

    return (
        <>
            <ListItemButton
                onClick={handleItemClick}
                onContextMenu={(e) => onContextMenu(e, item)}
                sx={{ pl: 2, borderRadius: 2 }}
                selected={isSelected} // Highlight the selected file
            >
                <ListItemIcon sx={{ minWidth: '32px', color: 'text.secondary' }}>
                    {item.type === 'folder' ? <FolderOpenIcon /> : <CodeIcon fontSize="small" />}
                </ListItemIcon>
                <ListItemText primary={item.name} />
                {item.type === 'folder' && (open ? <ExpandLess /> : <ExpandMore />)}
            </ListItemButton>
            {item.type === 'folder' && (
                <Collapse in={open} timeout="auto" unmountOnExit>
                    <List component="div" disablePadding sx={{ pl: 2 }}>
                        {item.children?.map(child => (
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
        </>
    );
};

export const ExplorerPanel: React.FC<{
      fileSystem: FileSystemItem[];
      onFileSelect: (fileId: string) => void;
      selectedFileId: string | null;
      onNewFile: (folderId?: string) => void;
      onNewFolder: (folderId?: string) => void;
      onDelete: (itemId: string) => void;
      onRename: (itemId: string, newName: string) => void;
  }> = ({ fileSystem, onFileSelect, selectedFileId, onNewFile, onNewFolder, onDelete, onRename }) => {

  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  
  // --- NEW: Context menu handler for the container/empty space ---
  const handleContainerContextMenu = (event: MouseEvent<HTMLDivElement>) => {
    // We only want this to fire on the container itself, not on the children.
    if (event.target !== event.currentTarget) {
        return;
    }
    event.preventDefault();
    setContextMenu({
        mouseX: event.clientX - 2,
        mouseY: event.clientY - 4,
        type: 'container', // Set the type to 'container'
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
    // This action now adds to the root level, which is correct for both menu types.
    onNewFile();
    handleClose();
  };
  
  const handleNewFolder = () => {
    onNewFolder();
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
          const newName = prompt("Enter new name:", contextMenu.item.name);
          if (newName) {
              onRename(contextMenu.item.id, newName);
          }
      }
      handleClose();
  };

  return (
    <Box sx={{ height: '100%', bgcolor: 'background.paper', p: 2, display: 'flex', flexDirection: 'column' }}>

      <Button variant="contained" color="primary" startIcon={<UploadFileIcon />} sx={{ mb: 2 }}>
        Import Strategies
      </Button>
      <Divider sx={{ my: 2 }} />
      
      <Typography variant="h2" sx={{ mb: 1, px: 1 }}>Strategy Explorer</Typography>
      
      {/* --- MODIFIED: Added onContextMenu to the List's parent Box --- */}
      <Box 
        onContextMenu={handleContainerContextMenu} 
        sx={{ flexGrow: 1 }} // Allow the box to fill the remaining space
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
        {/* --- MODIFIED: Render menu items based on the context menu 'type' --- */}
        {contextMenu?.type === 'container' && [
            <MenuItem key="new-file" onClick={handleNewFile}>
                <ListItemIcon><NoteAddIcon fontSize="small"/></ListItemIcon>
                <ListItemText>New File</ListItemText>
            </MenuItem>,
            <MenuItem key="new-folder" onClick={handleNewFolder}>
                <ListItemIcon><CreateNewFolderIcon fontSize="small"/></ListItemIcon>
                <ListItemText>New Folder</ListItemText>
            </MenuItem>,
            <MenuItem key="import" onClick={handleClose}>
                <ListItemIcon><UploadFileIcon fontSize="small"/></ListItemIcon>
                <ListItemText>Import Strategies</ListItemText>
            </MenuItem>
        ]}
        {contextMenu?.type === 'item' && contextMenu.item?.type === 'folder' && [
            <MenuItem key="new-file" onClick={handleNewFile}>New File</MenuItem>,
            <MenuItem key="new-folder" onClick={handleNewFolder}>New Folder</MenuItem>,
            <Divider key="divider" />,
        ]}
        {contextMenu?.type === 'item' && [
             <MenuItem key="rename" onClick={handleRename}>Rename</MenuItem>,
             <MenuItem key="delete" onClick={handleDelete} sx={{ color: 'error.main' }}>Delete</MenuItem>
        ]}
      </Menu>
    </Box>
  );
};