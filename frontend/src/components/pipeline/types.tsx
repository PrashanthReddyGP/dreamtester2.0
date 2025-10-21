// src/components/machinelearning/types.ts
export interface IndicatorParamDef {
    name: string;
    displayName: string;
    type: 'number' | 'string' | 'boolean';
    defaultValue: number | string | boolean;
    options?: string[];
}
export interface IndicatorDefinition {
    name: string;
    timeframe: string;
    params: IndicatorParamDef[];
}

export type IndicatorSchema = { [key: string]: IndicatorDefinition };

// Represents a single parameter for an indicator
export interface IndicatorParam {
  [key: string]: number | string;
}

// Represents a configured indicator instance
export interface IndicatorConfig {
    id: string;
    name: string;
    timeframe: string;
    params: IndicatorParam;
}

// The main configuration object for the entire ML pipeline
export interface MLConfig {
    problemDefinition: {
        type: 'template' | 'custom';
        templateKey: string;
        customCode: string;
    };
    dataSource: {
        symbol: string;
        timeframe: string;
        startDate: string;
        endDate: string;
    };
    features: IndicatorConfig[];
    model: {
        name: string;
        hyperparameters: {
            [key: string]: any;
        },
    };
    validation: {
        method: 'train_test_split' | 'walk_forward';
        trainSplit: number;
        walkForwardTrainWindow: number;
        walkForwardTestWindow: number;
    };
    preprocessing: {
        scaler: 'none' | 'StandardScaler' | 'MinMaxScaler';
        removeCorrelated: boolean;
        correlationThreshold: number;
        usePCA: boolean;
        pcaComponents: number;
        featureType: 'template' | 'custom';
        featureTemplateKey: string;
        customFeatureCode: string;
    };
    backtestSettings: {
        capital: number;
        risk: number;
        commissionBps: number;
        slippageBps: number;
        tradeOnClose: boolean;
    };
}