import React, { useState } from 'react';
import { Navigate, NavLink, useLocation } from 'react-router-dom';
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
import CandlestickChartIcon from '@mui/icons-material/CandlestickChart';
import { AnimatedPage } from '../components/common/AnimatedPage';
import { StrategyLab } from '../pages/StrategyLab';
import { ChartingView } from '../pages/ChartingView';
import { PipelineEditor } from '../pages/PipelineEditor';
import { MachineLearning } from '../pages/MachineLearning';
import { AnalysisHub } from '../pages/AnalysisHub';

interface MainLayoutProps {
  mode: 'light' | 'dark';
  toggleTheme: () => void;
}

// --- DEFINE YOUR PAGES HERE ---
const pages = [
  { path: '/lab', element: <StrategyLab />, nav: { text: 'Strategy Lab', icon: <ScienceIcon sx={{ mr: 1, fontSize: '1.25rem' }} /> } },
  { path: '/charting', element: <ChartingView />, nav: { text: 'Charting View', icon: <CandlestickChartIcon sx={{ mr: 1, fontSize: '1.25rem' }} /> } },
  { path: '/pipeline', element: <PipelineEditor />, nav: { text: 'Pipeline Editor', icon: <DesignServicesIcon sx={{ mr: 1, fontSize: '1.25rem' }} /> } },
  { path: '/machinelearning', element: <MachineLearning />, nav: { text: 'Machine Learning', icon: <PrecisionManufacturingIcon sx={{ mr: 1, fontSize: '1.25rem' }} /> } },
  { path: '/analysis', element: <AnalysisHub />, nav: { text: 'Analysis Hub', icon: <BarChartIcon sx={{ mr: 1, fontSize: '1.25rem' }} /> } },
  { path: '/automation', element: <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, textAlign: 'center', justifyContent: 'center' }}>Automation Page</div>, nav: { text: 'Automation', icon: <AutoAwesomeIcon sx={{ mr: 1, fontSize: '1.25rem' }} /> } },
];



export const MainLayout: React.FC<MainLayoutProps> = ({ mode, toggleTheme }) => {
  const theme = useTheme();
  const location = useLocation(); 
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { isTerminalOpen, toggleTerminal } = useTerminal();
  const [statusMessage] = useState('Ready'); // Placeholder state for the message
  
  const STATUS_BAR_HEIGHT = 24;

  // If the user lands on the root path, render ONLY the Navigate component.
  if (location.pathname === '/') {
    return <Navigate to="/lab" replace />;
  }

  const handleSettingsClick = () => setSettingsOpen(true);
  const VerticalResizeHandle = () => <PanelResizeHandle style={{ height: '1px', background: theme.palette.divider }} />;

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
              {pages.filter(p => p.nav).map((page) => (
                <Button
                  key={page.path}
                  component={NavLink}
                  to={page.path}
                  sx={{ color: 'text.secondary', fontWeight: 500, px: 2, py: 1, '&:hover': { backgroundColor: theme.palette.action.hover } }}
                  startIcon={page.nav!.icon}
                >
                  {page.nav!.text}
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
                {pages.map(page => (
                    <Box 
                        key={page.path} 
                        style={{ 
                            display: location.pathname === page.path ? 'flex' : 'none',
                            flexDirection: 'column',
                            flexGrow: 1,
                        }}
                    >
                        <AnimatedPage>{page.element}</AnimatedPage>
                    </Box>
                ))}
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