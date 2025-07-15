// src/services/websocketService.tsx

import { v4 as uuidv4 } from 'uuid';

// Define the shape of a listener function
// It receives the message data and has a unique ID for unsubscribing
type MessageListener = (data: any) => void;
interface Listener {
  id: string;
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
  listeners.forEach(listener => listener.callback(data));
};

/**
 * Adds a new listener for incoming messages.
 * @param callback The function to call when a message is received.
 * @returns An object with an `unsubscribe` function to remove the listener.
 */
const subscribe = (callback: MessageListener): { id: string; unsubscribe: () => void } => {
  const id = uuidv4();
  listeners.push({ id, callback });
  
  return {
    id,
    unsubscribe: () => {
      const index = listeners.findIndex(listener => listener.id === id);
      if (index > -1) {
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