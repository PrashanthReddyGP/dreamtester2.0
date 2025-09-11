import React, { useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { Box, CssBaseline, AppBar, Toolbar, Typography, IconButton, Button, useTheme, Paper } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import TerminalIcon from '@mui/icons-material/Terminal';
import BarChartIcon from '@mui/icons-material/BarChart';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import PrecisionManufacturingIcon from '@mui/icons-material/PrecisionManufacturing';
import SettingsIcon from '@mui/icons-material/Settings';
import { SettingsModal } from '../components/common/SettingsModal';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'; 
import { TerminalPanel } from '../components/common/TerminalPanel';
import { useTerminal } from '../context/TerminalContext';
import DesignServicesIcon from '@mui/icons-material/DesignServices';

interface MainLayoutProps {
  mode: 'light' | 'dark';
  toggleTheme: () => void;
}

const navItems = [
  { text: 'Strategy Lab', icon: <ScienceIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/lab' },
  { text: 'Pipeline Editor', icon: <DesignServicesIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/pipeline' },
  { text: 'Machine Learning', icon: <PrecisionManufacturingIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/machinelearning' },
  { text: 'Analysis Hub', icon: <BarChartIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/analysis' },
  { text: 'Automation', icon: <AutoAwesomeIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/automation' },
];

export const MainLayout: React.FC<MainLayoutProps> = ({ mode, toggleTheme }) => {
  const theme = useTheme();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { isTerminalOpen, toggleTerminal } = useTerminal();

  const STATUS_BAR_HEIGHT = 24;
  const [statusMessage, setStatusMessage] = useState('Ready'); // Placeholder state for the message

  // Style for active NavLink
  const activeLinkStyle = {
    backgroundColor: theme.palette.action.hover,
    color: theme.palette.text.primary,
  };
  

  // const handleToggleTerminal = () => {
  //   setIsTerminalOpen(prevState => !prevState);
  // };

  const handleSettingsClick = () => {
    setSettingsOpen(true);
  };

  // A simple vertical resize handle
  const VerticalResizeHandle = () => (
    <PanelResizeHandle style={{ height: '1px', background: theme.palette.divider }} />
  );

  return (
    <>
      <CssBaseline />
        <AppBar
          position="fixed"
          sx={{
            zIndex: (theme) => theme.zIndex.drawer + 1,
            backgroundColor: 'background.paper',
            borderBottom: 1,
            borderColor: 'divider',
          }}
          elevation={0}
        >
          <Toolbar>
            <Typography variant="h6" noWrap component="div" sx={{ color: 'text.primary', fontWeight: 'bold' }}>
              Dreamtester 2.0
            </Typography>
            <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', gap: 1 }}>
              {navItems.map((item) => (
                <Button
                  key={item.text}
                  component={NavLink}
                  to={item.path}
                  sx={{
                    color: 'text.secondary',
                    fontWeight: 500,
                    px: 2,
                    py: 1,
                    '&:hover': { backgroundColor: theme.palette.action.hover }
                  }}
                  startIcon={item.icon}
                >
                  {item.text}
                </Button>
              ))}
            </Box>
            <Box>

              <IconButton sx={{ ml: 1 }} onClick={toggleTheme} color="inherit">
                {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
              </IconButton>  

              <IconButton sx={{ ml: 1 }} onClick={() => toggleTerminal()} color="inherit">
                <TerminalIcon />
              </IconButton>

              <IconButton sx={{ ml: 1 }} color="inherit" onClick={handleSettingsClick}>
                <SettingsIcon />
              </IconButton>
            </Box>
          </Toolbar>
        </AppBar>

        <Box component="main" sx={{height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', bgcolor: 'background.default',}}>
          
          <Toolbar />
          
          <Box sx={{height: `calc(100vh - 64px - ${STATUS_BAR_HEIGHT}px)`}}>
          
            <PanelGroup direction="vertical">
          
              <Panel id='main-content' order={1} style={{display:'flex'}}>
                <Outlet />
              </Panel>

              {/* Conditionally Render the Terminal Panel and its Handle */}
              {isTerminalOpen && (
                <>
                  <VerticalResizeHandle />
                  <Panel id='terminal' order={2} defaultSize={25} minSize={15} maxSize={50}>
                    <TerminalPanel onClose={() => toggleTerminal(false)} />
                  </Panel>
                </>
              )}
          
            </PanelGroup>
          
          </Box>

          <Paper 
            component="footer" 
            square 
            elevation={0}
            sx={{
              height: `${STATUS_BAR_HEIGHT}px`,
              borderTop: 1,
              borderColor: 'divider',
              display: 'flex',
              alignItems: 'center',
              px: 2,
              gap: 1,
              backgroundColor: 'background.paper'
            }}
          >
            <CheckCircleOutlineIcon sx={{ color: 'success.main', fontSize: '1rem' }} />
            <Typography variant="caption" color="text.secondary">
              {statusMessage}
            </Typography>
          </Paper>

        </Box>

        <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
};