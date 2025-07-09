import React from 'react';
import { PanelResizeHandle } from 'react-resizable-panels';
import { Box } from '@mui/material';

export const ResizeHandle: React.FC = () => {
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