const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process'); // To run the Python script

let pythonProcess = null;
let mainWindow = null;

// --- Python Backend Handling ---
function createPythonProcess() {
    // Path to the Python executable inside the .venv
    // This is crucial for using the correct environment
    const pythonExe = path.join(__dirname, '..', 'backend', '.venv', 'Scripts', 'python.exe');
    const pythonScript = path.join(__dirname, '..', 'backend', 'main.py');

    // Spawn the Python process
    // We use 'uvicorn' to run the FastAPI app
    pythonProcess = spawn(pythonExe, [
        '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'
    ], {
        cwd: path.join(__dirname, '..', 'backend') // Set working directory
    });

    // Log output from the Python process for debugging
    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
    });
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
    });
    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}

function killPythonProcess() {
    if (pythonProcess) {
        console.log('Killing Python process...');
        pythonProcess.kill();
        pythonProcess = null;
    }
}

// --- Electron App Window ---
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 720,
        webPreferences: {
            // preload: path.join(__dirname, 'preload.js'), // Optional: for secure IPC
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    // Determine the URL to load
    const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, '../frontend/dist/index.html')}`;

    console.log(`Loading URL: ${startUrl}`);
    mainWindow.loadURL(startUrl);

    // To hide the default Electron Menubar
    mainWindow.removeMenu(); 

    // Open DevTools in development
    // if (process.env.ELECTRON_START_URL) {
    //     mainWindow.webContents.openDevTools();
    // }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.on('ready', () => {
    console.log('Starting Python backend...');
    createPythonProcess();
    createWindow();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', killPythonProcess);

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});