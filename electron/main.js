const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process'); // To run the Python script
const net = require('net'); // Import Node.js's net module

let pythonProcess = null;
let mainWindow = null;


// --- Function to check if a port is in use ---
function checkPortInUse(port, callback) {
    const server = net.createServer(function(socket) {
        socket.write('Echo server\r\n');
        socket.pipe(socket);
    });

    server.listen(port, '127.0.0.1');
    server.on('error', function (e) {
        // If we get an 'EADDRINUSE' error, it means the port is in use (success for us!)
        callback(true);
    });
    server.on('listening', function (e) {
        // If we can listen, it means the port is free.
        server.close();
        callback(false);
    });
}

// --- Function to poll for the backend server ---
function pollServer(port, callback) {
    let attempts = 0;
    const maxAttempts = 20; // Try for 10 seconds (20 * 500ms)
    
    function tryConnect() {
        console.log(`Checking if backend is up on port ${port} (Attempt ${attempts + 1})...`);
        checkPortInUse(port, (isInUse) => {
            if (isInUse) {
                console.log('Backend is up! Creating window.');
                callback();
            } else {
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(tryConnect, 500); // Wait 500ms and try again
                } else {
                    console.error('Backend server did not start in time.');
                    // You might want to show an error dialog to the user here
                    app.quit();
                }
            }
        });
    }
    
    tryConnect();
}

// --- Python Backend Handling ---
function createPythonProcess() {
    // Path to the Python executable inside the .venv
    // This is crucial for using the correct environment
    const pythonExe = path.join(__dirname, '..', 'backend', '.venv', 'Scripts', 'python.exe');
    const pythonScript = path.join(__dirname, '..', 'backend', 'main.py');

    const spawnArgs = ['-u', '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'];

    console.log('Spawning Python backend with command:', pythonExe, spawnArgs.join(' '));

    pythonProcess = spawn(pythonExe, spawnArgs, {
        cwd: path.join(__dirname, '..', 'backend'),
        stdio: 'pipe' // This is the key change!
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`[dev:electron] Python stdout: ${data.toString().trim()}`);
    });
    pythonProcess.stderr.on('data', (data) => {
        const stderrStr = data.toString().trim();
        if (stderrStr.includes('ERROR')) {
            console.error(`[dev:electron] Python stderr: ${stderrStr}`);
        } else {
            console.log(`[dev:electron] Python info: ${stderrStr}`);
        }
    });
    pythonProcess.on('close', (code) => {
        console.log(`[dev:electron] Python process exited with code ${code}`);
    });
}

function killPythonProcess() {
    if (pythonProcess) {
        console.log('Killing Python process...');
        // On Windows, 'SIGINT' (Ctrl+C) might not be enough.
        // 'taskkill' is a more reliable way to terminate a process tree.
        if (process.platform === 'win32') {
            const { exec } = require('child_process');
            exec(`taskkill /pid ${pythonProcess.pid} /f /t`, (error, stdout, stderr) => {
                if (error) {
                    console.error(`Error killing process: ${error.message}`);
                    return;
                }
                if (stderr) {
                    console.error(`Stderr on kill: ${stderr}`);
                    return;
                }
                console.log(`Successfully killed process: ${stdout}`);
            });
        } else {
            // The original kill for Mac/Linux
            pythonProcess.kill();
        }
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
    if (process.env.ELECTRON_START_URL) {
        mainWindow.webContents.openDevTools();
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.on('ready', () => {
    console.log('Starting Python backend...');
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