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
    socket.close();
  }

  // Create a new WebSocket connection
  socket = new WebSocket(`${WS_URL}${batchId}`);

  socket.onopen = () => {
    console.log(`WebSocket Service: Connection established for batch ${batchId}.`);
    // Notify all listeners that a connection is open
    broadcast({ type: 'system', payload: { event: 'open' } });
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      // Broadcast the parsed message to all active listeners
      broadcast(data);
    } catch (error) {
      console.error('WebSocket Service: Error parsing message', error);
      broadcast({ type: 'system', payload: { event: 'error', message: 'Failed to parse message' } });
    }
  };

  socket.onerror = (event) => {
    console.error('WebSocket Service: Error', event);
    broadcast({ type: 'system', payload: { event: 'error', message: 'WebSocket connection error' } });
  };

  socket.onclose = () => {
    console.log('WebSocket Service: Connection closed.');
    broadcast({ type: 'system', payload: { event: 'close' } });
    socket = null; // Clear the socket
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