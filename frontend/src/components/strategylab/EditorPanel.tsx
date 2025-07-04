import React from 'react';
import Editor from '@monaco-editor/react';
import { Box } from '@mui/material';

export const EditorPanel: React.FC<{ code: string | undefined }> = ({ code }) => {
  return (
      <Box sx={{ height: '100%', bgcolor: 'background.paper', p: 0.25 }}>
        <Editor
          height="100%"
          defaultLanguage="python"
          key={code}
          defaultValue={code}
          theme="app-dark-theme" 
          options={{
            minimap: { enabled: false},
            fontSize: 14,
            wordWrap: 'on',
            scrollBeyondLastLine: false,
            padding: { top: 24 },
            readOnly: false 
          }}
        />
      </Box>
  );
};