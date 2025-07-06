import React, { useState } from 'react';
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
const rsiMomentumCode = `print("This is the RSI Momentum strategy code.")`;
const smaCrossoverCode = `import crypto_backtester_core as core

class SmaCrossover(core.BaseStrategy):
    """
    A simple moving average crossover strategy.
    """
    def init(self):
        self.fast_sma_period = 20
        self.slow_sma_period = 50
    
    def next(self, current_bar_index):
        # Your logic here
        pass
`;

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
    // --- 2. LIFTED STATE: State for the file system and selected file ID ---
    const [fileSystem, setFileSystem] = useState<FileSystemItem[]>(initialFileSystem);
    const [selectedFileId, setSelectedFileId] = useState<string | null>('sma-crossover-id'); // Select one by default

    const handleFileSelect = (fileId: string) => {setSelectedFileId(fileId);};
    const handleNewFile = (folderId?: string | undefined) => {};
    const handleNewFolder = (folderId?: string | undefined) => {};
    const handleDelete = (fileId: string) => {setSelectedFileId(fileId);};
    const handleRename = (fileId: string) => {setSelectedFileId(fileId);};

    // --- 3. Find the currently selected file's content ---
    const selectedFile = selectedFileId ? findFileById(fileSystem, selectedFileId) : null;
    const editorCode = selectedFile ? selectedFile.content : '// Select a file from the explorer to view its content.';

  return (
    <Box sx={{ height: '100%', width: '100vw' }}>
      
      <PanelGroup direction="horizontal">

        <Panel defaultSize={15} minSize={15}>
            <ExplorerPanel
                fileSystem={fileSystem}
                onFileSelect={handleFileSelect}
                selectedFileId={selectedFileId}
                onNewFile={handleNewFile}
                onNewFolder={handleNewFolder}
                onDelete={handleDelete}
                onRename={handleRename}
            />
        </Panel>

        <ResizeHandle />

        <Panel defaultSize={70} minSize={40}>
            <EditorPanel code={editorCode} />
        </Panel>

        <ResizeHandle />

        <Panel defaultSize={15} minSize={15}>
          <SettingsPanel />
        </Panel>
        
      </PanelGroup>
    </Box>
  );
};