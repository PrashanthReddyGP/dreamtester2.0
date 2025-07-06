import React, { useState, useRef, forwardRef } from 'react';
import type { MouseEvent, FC } from 'react';
import { Box, Button, Collapse, Divider, List, ListItemButton, ListItemIcon, ListItemText, Typography, Menu, MenuItem } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import NoteAddIcon from '@mui/icons-material/NoteAdd';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import { Folder as FolderIcon, FolderOpen, File, Trash } from 'lucide-react';

import { useDraggable, useDroppable } from '@dnd-kit/core';

// --- Data structure remains the same ---
export interface FileSystemItem {
  id: string;
  name: string;
  type: 'folder' | 'file';
  children?: FileSystemItem[];
  content?: string;
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
      onImportFile: (name: string, content: string) => void;
      onClearAll: () => void;
  }> = ({ fileSystem, onFileSelect, selectedFileId, onNewFile, onNewFolder, onDelete, onRename, onImportFile, onClearAll }) => {
  
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

    const file = files[0];
    const reader = new FileReader();

    reader.onload = (e) => {
      const content = e.target?.result;
      if (typeof content === 'string') {
        onImportFile(file.name, content);
      }
    };

    reader.onerror = (e) => {
      console.error("Error reading file:", e);
      alert("Failed to read the selected file.");
    };

    reader.readAsText(file);

    // Reset the input value so the onChange event fires again for the same file
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
      />

      <Divider sx={{ my: 2 }} />
      
      <Typography variant="h2" sx={{ mb: 1, px: 1 }}>Strategy Explorer</Typography>
      
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
             <MenuItem key="delete" onClick={handleDelete} sx={{ color: 'error.main' }}>Delete</MenuItem>
        ]}
      </Menu>
    </Box>
  );
};