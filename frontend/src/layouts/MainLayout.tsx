import React, { useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { Box, CssBaseline, AppBar, Toolbar, Typography, IconButton, Button, useTheme } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import BarChartIcon from '@mui/icons-material/BarChart';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import SettingsIcon from '@mui/icons-material/Settings';
import { SettingsModal } from '../components/common/SettingsModal';

interface MainLayoutProps {
  mode: 'light' | 'dark';
  toggleTheme: () => void;
}

const navItems = [
  { text: 'Strategy Lab', icon: <ScienceIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/lab' },
  { text: 'Analysis Hub', icon: <BarChartIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/analysis' },
  { text: 'Automation', icon: <AutoAwesomeIcon sx={{ mr: 1, fontSize: '1.25rem' }} />, path: '/automation' },
];

export const MainLayout: React.FC<MainLayoutProps> = ({ mode, toggleTheme }) => {
  const theme = useTheme();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleSettingsClick = () => {
    setSettingsOpen(true);
  };

  // Style for active NavLink
  const activeLinkStyle = {
    backgroundColor: theme.palette.action.hover,
    color: theme.palette.text.primary,
  };

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
                style={({ isActive }) => (isActive ? activeLinkStyle : {})}
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
            <IconButton sx={{ ml: 1 }} color="inherit" onClick={handleSettingsClick}>
              <SettingsIcon />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Box
        component="main"
        sx={{
          height: '100vh',
          width: '100vw',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: 'background.default',
          // overflow: 'clip'
        }}
      >
        <Toolbar />
        
        <Box sx={{ flexGrow: 1, display: 'flex'}}>
          <Outlet />
        </Box>
      </Box>

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
};