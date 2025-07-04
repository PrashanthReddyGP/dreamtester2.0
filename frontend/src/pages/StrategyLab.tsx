import React from 'react';
import { Box } from '@mui/material';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";

import { ExplorerPanel } from '../components/strategylab/ExplorerPanel';
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

export const StrategyLab: React.FC = () => {
  return (
    <Box sx={{ height: '100%', width: '100vw' }}>
      <PanelGroup direction="horizontal">

        <Panel defaultSize={15} minSize={15}>
          <ExplorerPanel />
        </Panel>

        <ResizeHandle />

        <Panel defaultSize={70} minSize={40}>
          <EditorPanel />
        </Panel>

        <ResizeHandle />

        <Panel defaultSize={15} minSize={15}>
          <SettingsPanel />
        </Panel>
        
      </PanelGroup>
    </Box>
  );
};