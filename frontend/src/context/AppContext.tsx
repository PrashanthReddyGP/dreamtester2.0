import React, { createContext, useContext } from 'react';
import type { AppContextType } from './types';

// Define the shape of the context data for clarity and autocompletion
const defaultAppContext: AppContextType = {
  settings: {}, // This empty object is now correctly typed as SettingsState
  isLoading: true,
  error: null,
  saveApiKeys: async (exchange, apiKey, apiSecret) => { 
    console.error("saveApiKeys function not yet implemented");
    return Promise.reject("Not implemented");
  },
};

const AppContext = createContext<AppContextType>(defaultAppContext);

// Create a custom hook for easy consumption of the context in other components
export const useAppContext = () => {
  return useContext(AppContext);
};

export default AppContext;