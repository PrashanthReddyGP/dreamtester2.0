// src/services/websocketService.tsx

import { v4 as uuidv4 } from 'uuid';

// Define the shape of a listener function
// It receives the message data and has a unique ID for unsubscribing
type MessageListener = (data: any) => void;
interface Listener {
  key: string; 
  callback: MessageListener;
}

const WS_URL = 'ws://127.0.0.1:8000/ws/results/';
let socket: WebSocket | null = null;
const listeners: Listener[] = [];

/**
 * Connects to the WebSocket server for a given batch ID.
 * Closes any existing connection before opening a new one.
 * @param batchId The unique ID for the backtest batch.
 */
const connect = (batchId: string): void => {
  // If there's an existing connection, close it.
  if (socket) {
    console.log("WebSocket Service: Closing existing socket before reconnecting.");
    socket.close();
  }
  
  // A new connection means a new session. Clear out any old listeners
  // that might be lingering from a previous session or a hot-reload.
  // listeners.length = 0;
  // console.log("%cWebSocket Service: Purged all listeners for new session.", 'color: orange; font-weight: bold;');

  const url = `${WS_URL}${batchId}`;
  console.log(`%cWebSocket Service: Attempting to connect to ${url}`, 'color: blue; font-weight: bold;');
  socket = new WebSocket(url);

  socket.onopen = (event) => {
    console.log(`%cWebSocket Service: ONOPEN - Connection established.`, 'color: green; font-weight: bold;');
    broadcast({ type: 'system', payload: { event: 'open' } });
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      broadcast(data);
    } catch (error) {
      console.error('WebSocket Service: Error parsing message', error);
      broadcast({ type: 'system', payload: { event: 'error', message: 'Failed to parse message' } });
    }
  };

  socket.onerror = (event) => {
    console.error(`%cWebSocket Service: ONERROR - An error occurred.`, 'color: red; font-weight: bold;', event);
    broadcast({ type: 'system', payload: { event: 'error', message: 'WebSocket connection error' } });
  };

  socket.onclose = (event) => {
    console.error(`%cWebSocket Service: ONCLOSE - Connection closed.`, 'color: red; font-weight: bold;');
    console.log(`- Was clean: ${event.wasClean}`);
    console.log(`- Code: ${event.code}`);
    console.log(`- Reason: "${event.reason}"`);
    broadcast({ type: 'system', payload: { event: 'close' } });
    socket = null;
  };
};

/**
 * Sends a message to all registered listeners.
 * @param data The data to send.
 */
const broadcast = (data: any): void => {
  // Create a copy to prevent issues if a listener unsubscribes during the loop
  const listenersCopy = [...listeners]; 
  
  listenersCopy.forEach(listener => {
    // --- ROBUSTNESS CHECK ---
    // Check if the callback is actually a function before trying to call it.
    if (typeof listener.callback === 'function') {
      listener.callback(data);
    } else {
      // This will now log the exact problem instead of crashing the app.
      console.error('WebSocket Service Broadcast Error: Found a listener whose callback is not a function!', listener);
    }
  });
};

/**
 * Adds or updates a listener for a given key. Ensures no duplicates.
 * @param key A unique string identifying the listener (e.g., 'terminal-context').
 * @param callback The function to call when a message is received.
 * @returns An object with an `unsubscribe` function to remove the listener.
 */
const subscribe = (key: string, callback: MessageListener): { unsubscribe: () => void } => {
  const existingListenerIndex = listeners.findIndex(listener => listener.key === key);

  if (existingListenerIndex > -1) {
    // If a listener with this key already exists, just update its callback.
    // This handles the HMR case where the component re-renders with a new function instance.
    console.warn(`WebSocket Service: Updating listener for key "${key}".`);
    listeners[existingListenerIndex].callback = callback;
  } else {
    // Otherwise, add a new listener.
    console.log(`WebSocket Service: Adding new listener for key "${key}".`);
    listeners.push({ key, callback });
  }
  
  // The unsubscribe function now uses the key to find and remove the listener.
  return {
    unsubscribe: () => {
      const index = listeners.findIndex(listener => listener.key === key);
      if (index > -1) {
        console.log(`WebSocket Service: Unsubscribing listener for key "${key}".`);
        listeners.splice(index, 1);
      }
    },
  };
};


/**
 * Closes the WebSocket connection if it's open.
 */
const disconnect = (): void => {
  if (socket) {
    socket.close();
  }
};

// Export the public methods of the service
export const websocketService = {
  connect,
  disconnect,
  subscribe,
};