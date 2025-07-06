// src/context/types.ts

export interface ApiKeySet {
  exchange: string; // <-- ADD THIS
  apiKey: string;
  apiSecret: string;
}

// ... rest of the file is the same ...
export interface SettingsState {
  [exchange: string]: Omit<ApiKeySet, 'exchange'> | undefined; // Use Omit to not store redundant exchange name
}

export interface AppContextType {
  settings: SettingsState;
  isLoading: boolean;
  error: string | null;
  saveApiKeys: (exchange: string, apiKey: string, apiSecret: string) => Promise<any>;
}