// electron/main.js

const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');

let pythonProcess = null;
let mainWindow = null;

// --- No changes needed for checkPortInUse or pollServer ---
function checkPortInUse(port, callback) {
    const server = net.createServer(function(socket) {
        socket.write('Echo server\r\n');
        socket.pipe(socket);
    });
    server.listen(port, '127.0.0.1');
    server.on('error', function (e) { callback(true); });
    server.on('listening', function (e) { server.close(); callback(false); });
}

function pollServer(port, callback) {
    let attempts = 0;
    const maxAttempts = 20;
    function tryConnect() {
        console.log(`Checking if backend is up on port ${port} (Attempt ${attempts + 1})...`);
        checkPortInUse(port, (isInUse) => {
            if (isInUse) {
                console.log('Backend is up! Creating window.');
                callback();
            } else {
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(tryConnect, 500);
                } else {
                    console.error('Backend server did not start in time.');
                    app.quit();
                }
            }
        });
    }
    tryConnect();
}


// --- THIS FUNCTION IS MODIFIED FOR CORRECT PATHS ---
function createPythonProcess() {
    const isDev = !app.isPackaged;

    // `__dirname` is ".../Dreamtester_2.0/electron"
    // We go UP one level to the project root.
    const projectRoot = path.join(__dirname, '..'); 

    // In production, the backend is in 'resources/backend'
    // In development, it's in './backend' relative to the project root.
    const backendPath = isDev 
      ? path.join(projectRoot, 'backend')
      : path.join(process.resourcesPath, 'backend');

    const pythonExe = path.join(backendPath, '.venv', 'Scripts', 'python.exe');
    const scriptCwd = backendPath;
    const spawnArgs = ['-u', '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'];

    console.log('--- Spawning Python Backend ---');
    console.log(`Mode: ${isDev ? 'Development' : 'Production'}`);
    console.log(`Executable: ${pythonExe}`);
    console.log(`CWD: ${scriptCwd}`);
    console.log('-----------------------------');

    pythonProcess = spawn(pythonExe, spawnArgs, {
        cwd: scriptCwd,
        stdio: 'pipe'
    });

    pythonProcess.stdout.on('data', (data) => console.log(`[Python] ${data.toString().trim()}`));
    pythonProcess.stderr.on('data', (data) => console.error(`[Python stderr] ${data.toString().trim()}`));
    pythonProcess.on('close', (code) => console.log(`Python process exited with code ${code}`));
}

// --- No changes needed for killPythonProcess ---
function killPythonProcess() {
    if (pythonProcess) {
        console.log('Killing Python process...');
        if (process.platform === 'win32') {
            const { exec } = require('child_process');
            exec(`taskkill /pid ${pythonProcess.pid} /f /t`, (err, stdout, stderr) => { /* ... */ });
        } else {
            pythonProcess.kill();
        }
        pythonProcess = null;
    }
}

// --- THIS FUNCTION IS MODIFIED FOR CORRECT PATHS ---
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 720,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    // In dev, the Vite server provides a URL.
    // In production, we need to load the built frontend file.
    // The path is relative to the project root, so we go up from __dirname.
    const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, '..', 'frontend', 'dist', 'index.html')}`;

    console.log(`Loading URL: ${startUrl}`);
    mainWindow.loadURL(startUrl);

    mainWindow.removeMenu(); 

    if (process.env.ELECTRON_START_URL) {
        mainWindow.webContents.openDevTools();
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// --- No changes needed for app lifecycle events ---
app.on('ready', () => {
    console.log('App ready, starting backend...');
    createPythonProcess();
    pollServer(8000, createWindow);
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