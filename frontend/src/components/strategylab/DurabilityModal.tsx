import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Box,
    TextField,
    Typography,
    Checkbox,
    FormControlLabel,
    CircularProgress,
    Alert,
    IconButton,
    Divider,
    Autocomplete,
    Chip,
    ToggleButtonGroup,
    ToggleButton,
    Select,
    MenuItem,
    Slider
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { fetchAvailableSymbols } from '../../services/api';
import { v4 as uuidv4 } from 'uuid'; // For unique rule IDs
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';

// This should be your actual backend URL
const API_URL = 'http://127.0.0.1:8000'; 

interface OptimizableParameter {
    id: string;
    type: 'strategy_param' | 'indicator_param';
    name: string;
    value: number;
    enabled: boolean;
    mode: 'range' | 'list';
    start: number;
    end: number;
    step: number;
    list_values: string;
    indicatorIndex?: number;
    paramIndex?: number;
}

export interface CombinationRule {
    id: string;
    param1: string;
    operator: '<' | '>' | '<=' | '>=' | '===' | '!==';
    param2: string;
}


// This defines the base data sent for any test.
export interface BaseDurabilityConfig {
    strategy_code: string;
    parameters_to_optimize: OptimizableParameter[];
    symbols_to_screen: string[];
    combination_rules: Omit<CombinationRule, 'id'>[];
}

// This defines the specific parameters for a Data Segmentation test.
export interface DataSegmentationDetails {
    test_type: 'data_segmentation';
    training_pct: number;
    validation_pct: number;
    testing_pct: number;
    optimization_metric: string;
    top_n_sets: number;
}

export interface WalkForwardDetails {
    test_type: 'walk_forward';
    training_period_length: number;
    training_period_unit: 'days' | 'weeks' | 'months';
    testing_period_length: number;
    testing_period_unit: 'days' | 'weeks' | 'months';
    is_anchored: boolean;
    step_forward_pct: number;
    optimization_metric: string;
}

// We can add other test types here in the future
// export interface MonteCarloDetails { test_type: 'monte_carlo'; ... }

// The final submission object is a combination of the base config and one of the specific test details.
export type DurabilitySubmissionConfig = BaseDurabilityConfig & (DataSegmentationDetails | WalkForwardDetails /* | MonteCarloDetails */);


// The props for the modal now expect the new config type
interface DurabilityModalProps {
    open: boolean;
    onClose: () => void;
    onSubmit: (config: DurabilitySubmissionConfig) => void; 
    strategyCode: string | null;
    isSubmitting: boolean;
}

// Define the available test modes
type DurabilityTestMode = 'monte_carlo' | 'data_segmentation' | 'walk_forward' | 'correlated_assets' | 'parameter_range' | 'market_regime';


const fetchAllParametersAndSettings = async (code: string): Promise<any> => {
    const response = await fetch(`${API_URL}/api/strategies/parse-parameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: code,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "An unknown error" }));
        throw new Error(errorData.detail);
    }
    return response.json();
};


interface CommonPanelsProps {
    symbolList: string[];
    selectedSymbols: string[];
    setSelectedSymbols: (symbols: string[]) => void;
    isFetchingSymbols: boolean;
    rules: CombinationRule[];
    enabledParamsForRules: OptimizableParameter[];
    addRule: () => void;
    updateRule: (id: string, field: keyof Omit<CombinationRule, 'id'>, value: string) => void;
    removeRule: (id: string) => void;
    params: OptimizableParameter[];
    handleParamChange: (id: string, field: keyof OptimizableParameter, value: any) => void;
    isSubmitting: boolean;
}

const CommonPanels: React.FC<CommonPanelsProps> = ({
    symbolList,
    selectedSymbols,
    setSelectedSymbols,
    isFetchingSymbols,
    rules,
    enabledParamsForRules,
    addRule,
    updateRule,
    removeRule,
    params,
    handleParamChange,
    isSubmitting
}) => {
    return (
        <>
            {/* --- LEFT PANEL: ASSET SELECTION --- */}
            <Box sx={{ flex: 1, minWidth: '300px', gap: 3, display: 'flex', flexDirection: 'column' }}>
                <Box sx={{flexGrow: 1}}>
                    <Typography variant="h6" gutterBottom>Select Assets</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                    Choose one or more assets to run the test against.
                    </Typography>
                    <Autocomplete
                        multiple id="asset-screener-autocomplete" options={symbolList} value={selectedSymbols} loading={isFetchingSymbols}
                        onChange={(event, newValue) => { setSelectedSymbols(newValue); }}
                        freeSolo
                        renderTags={(value, getTagProps) => value.map((option, index) => {
                            const { key, ...tagProps } = getTagProps({ index });
                            return <Chip key={key} variant="outlined" label={option} {...tagProps} />;
                        })}
                        renderInput={(params) => (<TextField {...params} variant="outlined" label="Symbols to Test" placeholder="Add symbols..."/>)}
                    />
                </Box>

                <Divider orientation="horizontal" flexItem />

                <Box sx={{flexGrow: 1}}>
                    <Typography variant="h6" gutterBottom>Define Combination Rules</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                    Prevent illogical tests. For example, ensure a "Fast" SMA period is always less than a "Slow" SMA period.
                    </Typography>
                    
                    {rules.map((rule) => (
                    <CombinationRuleRow 
                        key={rule.id} 
                        rule={rule} 
                        availableParams={enabledParamsForRules}
                        onUpdate={updateRule}
                        onRemove={removeRule}
                    />
                    ))}

                    <Button
                        startIcon={<AddCircleOutlineIcon />}
                        onClick={addRule}
                        disabled={enabledParamsForRules.length < 2}
                        sx={{mt: 1}}
                    >
                        Add Rule
                    </Button>
                </Box>
            </Box>

            <Divider orientation="vertical" flexItem />

            {/* --- RIGHT PANEL: PARAMETER OPTIMIZATION (THE ONE WITH THE BUG) --- */}
            <Box sx={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
                <Typography variant="h6" gutterBottom>Configure Parameters</Typography>
                <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                    Enable and configure any parameters you wish to optimize. If none are enabled, the default values from the file will be used.
                </Typography>
                <Box sx={{maxHeight: '100%', overflowY: 'auto'}}>
                    {params.filter(p => p.type === 'strategy_param').length > 0 && <Typography variant="overline">Strategy Parameters</Typography>}
                    {params.filter(p => p.type === 'strategy_param').map(param => (
                        <ParameterInputRow key={param.id} param={param} handleParamChange={handleParamChange} isSubmitting={isSubmitting} />
                    ))}
                    
                    {params.filter(p => p.type === 'indicator_param').length > 0 && <Typography variant="overline" sx={{mt:2}}>Indicator Parameters</Typography>}
                    {params.filter(p => p.type === 'indicator_param').map(param => (
                        <ParameterInputRow key={param.id} param={param} handleParamChange={handleParamChange} isSubmitting={isSubmitting} />
                    ))}
                </Box>
            </Box>
        </>
    );
};


// --- The Main Modal Component ---
export const DurabilityModal: React.FC<DurabilityModalProps> = ({ open, onClose, onSubmit, strategyCode, isSubmitting }) => {
    const [params, setParams] = useState<OptimizableParameter[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [rules, setRules] = useState<CombinationRule[]>([]);

    const [mode, setMode] = useState<DurabilityTestMode>('data_segmentation');

    const [symbolList, setSymbolList] = useState<string[]>([]);
    const [isFetchingSymbols, setIsFetchingSymbols] = useState(false);
    const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
    const [defaultSymbol, setDefaultSymbol] = useState<string | null>(null);

    // State for Data Segmentation sliders
    const [trainingPct, setTrainingPct] = useState(60);
    const [validationPct, setValidationPct] = useState(20);
    const [testingPct, setTestingPct] = useState(20);

    const [optimizationMetric, setOptimizationMetric] = useState('Equity_Efficiency_Rate');
    const [topParamSets, setTopParamSets] = useState(10);

    // --- NEW: State for Walk-Forward Analysis ---
    const [trainingPeriodLength, setTrainingPeriodLength] = useState(180);
    const [trainingPeriodUnit, setTrainingPeriodUnit] = useState<'days' | 'weeks' | 'months'>('days');
    const [testingPeriodLength, setTestingPeriodLength] = useState(60);
    const [testingPeriodUnit, setTestingPeriodUnit] = useState<'days' | 'weeks' | 'months'>('days');
    const [isAnchored, setIsAnchored] = useState(false);
    const [stepForwardPct, setStepForwardPct] = useState(25);

    const hasFetchedData = React.useRef(false);

    useEffect(() => {
        // If the modal is not open, reset the fetch tracker.
        if (!open) {
        hasFetchedData.current = false;
        return;
        }

        // Only fetch data if the modal is open AND we haven't fetched it yet.
        if (open && strategyCode && !hasFetchedData.current) {
        setIsLoading(true);
        setError(null);

      // Fetch both symbols and parameters at the same time.
        Promise.all([
            fetchAvailableSymbols('binance'),
            fetchAllParametersAndSettings(strategyCode)
        ]).then(([fetchedSymbols, parsedData]) => {
            // --- Handle Symbols ---
            setSymbolList(fetchedSymbols);
            const symbolFromFile = parsedData.settings?.symbol;
            if (symbolFromFile) {
                setDefaultSymbol(symbolFromFile);
                setSelectedSymbols([symbolFromFile]);
            } else {
                setSelectedSymbols([]);
            }

            // --- Handle Parameters ---
            const optimizableParams = parsedData.optimizable_params.map((p: any): OptimizableParameter => ({
                ...p,
                enabled: false,
                mode: 'range', // Default to 'range' mode
                start: p.value,
                end: p.value,
                step: 1,
                list_values: String(p.value), // Default list is the single default value
            }));
            setParams(optimizableParams);

            // Mark that we have successfully fetched the data.
            hasFetchedData.current = true;
        
        }).catch(err => {
            setError(err.message);
        }).finally(() => {
            setIsLoading(false);
        });
    }
  }, [open, strategyCode]); // This effect now correctly depends only on `open` and `strategyCode`.

    const handleParamChange = (id: string, field: keyof OptimizableParameter, value: any) => {
        setParams(prev => prev.map(p => p.id === id ? { ...p, [field]: value } : p));
    };

    const handleModeChange = (event: React.MouseEvent<HTMLElement>, newMode: DurabilityTestMode | null) => {
        if (newMode !== null) {
            setMode(newMode);
        }
    };

    const handleSubmit = () => {
        if (!strategyCode) {
            setError("Strategy code is missing."); 
            return;
        }
        if (selectedSymbols.length === 0) {
            setError("Please select at least one asset to test."); 
            return;
        }

        // 1. Gather the common configuration data
        const baseConfig: BaseDurabilityConfig = {
            strategy_code: strategyCode,
            parameters_to_optimize: params.filter(p => p.enabled),
            symbols_to_screen: selectedSymbols,
            combination_rules: rules.map(({ id, ...rest }) => rest),
        };

        // 2. Depending on the mode, add the specific configuration and submit
        if (mode === 'data_segmentation') {
            const config: DurabilitySubmissionConfig = {
                ...baseConfig,
                test_type: 'data_segmentation',
                training_pct: trainingPct,
                validation_pct: validationPct,
                testing_pct: testingPct,
                optimization_metric: optimizationMetric,
                top_n_sets: topParamSets,
            };
            onSubmit(config);
        } else if (mode === 'walk_forward') { // --- NEW: Handle Walk-Forward submission ---
            const config: DurabilitySubmissionConfig = {
                ...baseConfig,
                test_type: 'walk_forward',
                training_period_length: trainingPeriodLength,
                training_period_unit: trainingPeriodUnit,
                testing_period_length: testingPeriodLength,
                testing_period_unit: testingPeriodUnit,
                is_anchored: isAnchored,
                step_forward_pct: stepForwardPct,
                optimization_metric: optimizationMetric,
            };
            onSubmit(config);
        } else if (mode === 'monte_carlo') {
            // Placeholder for when you implement Monte Carlo
            console.log("Submitting Monte Carlo Test (not yet fully implemented)");
            // Example structure:
            // const config = { ...baseConfig, test_type: 'monte_carlo', runs: 100, ... };
            // onSubmit(config);
        }
        // ... add other `else if` blocks for other modes
        else {
            setError(`Submission for test type "${mode}" is not implemented yet.`);
        }
    };
    

    const addRule = () => {
        setRules([...rules, { id: uuidv4(), param1: '', operator: '<', param2: '' }]);
    };
    const updateRule = (id: string, field: keyof Omit<CombinationRule, 'id'>, value: string) => {
        setRules(rules.map(rule => (rule.id === id ? { ...rule, [field]: value } : rule)));
    };
    const removeRule = (id: string) => {
        setRules(rules.filter(rule => rule.id !== id));
    };

    // Get a list of enabled parameters for the rule dropdowns
    const enabledParamsForRules = params.filter(p => p.enabled);

    const commonPanelProps = {
        symbolList,
        selectedSymbols,
        setSelectedSymbols,
        isFetchingSymbols,
        rules,
        enabledParamsForRules,
        addRule,
        updateRule,
        removeRule,
        params,
        handleParamChange,
        isSubmitting,
    };


return (
    <Dialog className='durability-panel' open={open} onClose={onClose} sx={{ display:'flex', justifyContent:'center', backdropFilter: 'blur(4px)',
            '& .css-iuy973-MuiPaper-root-MuiDialog-paper': {
                width: '90vw',
                height: '90vh',
                maxWidth: '100%',
            }
        }}>
        
        <DialogTitle sx={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
            Durability Test Configuration
            <>
                <ToggleButtonGroup
                color="primary"
                value={mode}
                exclusive
                onChange={handleModeChange}
                aria-label="Durability Test Mode"
                >
                <ToggleButton value="monte_carlo">Monte Carlo Simulation</ToggleButton>
                <ToggleButton value="data_segmentation">Data Segmentation</ToggleButton>
                <ToggleButton value="walk_forward">Walk-Forward Analysis</ToggleButton>
                <ToggleButton value="correlated_assets">Correlated Assets Test</ToggleButton>
                <ToggleButton value="parameter_range">Parameter Space Stability</ToggleButton>
                <ToggleButton value="market_regime">Market Regime Analysis</ToggleButton>
                </ToggleButtonGroup>
            <IconButton aria-label="close" onClick={onClose}><CloseIcon/></IconButton>
            </>
        </DialogTitle>

        <DialogContent dividers sx={{backgroundColor: 'background.default', height: '100%', ml:1, mr:1}}>
            {isLoading && (<Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /><Typography sx={{ml:2}}>Fetching Initial Data...</Typography></Box>)}
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            
            {!isLoading && !error && (
            <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
                {mode === 'parameter_range' && (
                <Box sx={{ display: 'flex', gap: 3, height: '100%' }}>
                    {/* --- LEFT PANEL: ASSET SELECTION --- */}
                    <Box sx={{ flex: 1, minWidth: '300px', gap: 3, display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{flexGrow: 1}}>
                        <Typography variant="h6" gutterBottom>Select Assets</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                        Choose one or more assets to run the test against.
                        </Typography>
                        <Autocomplete
                        multiple id="asset-screener-autocomplete" options={symbolList} value={selectedSymbols} loading={isFetchingSymbols}
                        onChange={(event, newValue) => { setSelectedSymbols(newValue); }}
                        freeSolo
                        renderTags={(value, getTagProps) => value.map((option, index) => {
                            const { key, ...tagProps } = getTagProps({ index });
                            return <Chip key={key} variant="outlined" label={option} {...tagProps} />;
                        })}
                        renderInput={(params) => (<TextField {...params} variant="outlined" label="Symbols to Test" placeholder="Add symbols..."/>)}
                        />
                    </Box>

                    <Divider orientation="horizontal" flexItem />

                    <Box sx={{flexGrow: 1}}>
                        <Typography variant="h6" gutterBottom>Define Combination Rules</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                        Prevent illogical tests. For example, ensure a "Fast" SMA period is always less than a "Slow" SMA period.
                        </Typography>
                        
                        {rules.map((rule) => (
                        <CombinationRuleRow 
                            key={rule.id} 
                            rule={rule} 
                            availableParams={enabledParamsForRules}
                            onUpdate={updateRule}
                            onRemove={removeRule}
                        />
                        ))}

                        <Button
                        startIcon={<AddCircleOutlineIcon />}
                        onClick={addRule}
                        disabled={enabledParamsForRules.length < 2}
                        sx={{mt: 1}}
                        >
                        Add Rule
                        </Button>
                    </Box>
                    </Box>

                    <Divider orientation="vertical" flexItem />

                    {/* --- RIGHT PANEL: PARAMETER OPTIMIZATION --- */}
                    <Box sx={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
                        <Typography variant="h6" gutterBottom>Configure Parameters</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                            Enable and configure any parameters you wish to optimize. If none are enabled, the default values from the file will be used.
                        </Typography>
                        <Box sx={{maxHeight: '100%', overflowY: 'auto'}}>
                            {params.filter(p => p.type === 'strategy_param').length > 0 && <Typography variant="overline">Strategy Parameters</Typography>}
                            {params.filter(p => p.type === 'strategy_param').map(param => (
                            <ParameterInputRow key={param.id} param={param} handleParamChange={handleParamChange} isSubmitting={isSubmitting} />
                            ))}
                            
                            {params.filter(p => p.type === 'indicator_param').length > 0 && <Typography variant="overline" sx={{mt:2}}>Indicator Parameters</Typography>}
                            {params.filter(p => p.type === 'indicator_param').map(param => (
                            <ParameterInputRow key={param.id} param={param} handleParamChange={handleParamChange} isSubmitting={isSubmitting} />
                            ))}
                        </Box>
                    </Box>
                </Box>
                )}
                {mode === 'monte_carlo' && (
                <Box sx={{ p: 3, display: 'flex', flexDirection: 'row', gap: 4, height: '100%' }}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4, borderRight: 1, pr: 4, borderColor: 'divider' }}>
                        <Box>
                            <Typography variant="h6">Monte Carlo Simulation</Typography>
                            <Typography variant="body2" color="text.secondary">
                                Test the strategy's robustness by introducing randomness to trade history and parameters.
                            </Typography>
                        </Box>

                        {/* Simulation Settings */}
                        <Box>
                            <Typography variant="subtitle1" gutterBottom sx={{ pb: 2 }}>Simulation Settings</Typography>
                            <TextField
                                fullWidth
                                type="number"
                                label="Number of Simulation Runs"
                                defaultValue={100}
                                helperText="The number of random scenarios to generate and test."
                                variant="outlined"
                                InputProps={{ inputProps: { min: 10, max: 10000 } }}
                            />
                        </Box>

                        {/* Trade Manipulation */}
                        <Box>
                            <Typography variant="subtitle1" gutterBottom>Trade Manipulation</Typography>
                            <FormControlLabel
                                control={<Checkbox defaultChecked />}
                                label="Shuffle the order of trades"
                            />
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>
                                Randomly skip a percentage of trades
                            </Typography>
                            <Slider
                                defaultValue={10}
                                aria-label="Trade Skip Percentage"
                                valueLabelDisplay="auto"
                                step={1}
                                marks
                                min={0}
                                max={50}
                            />
                        </Box>

                        {/* Parameter Randomization */}
                        <Box>
                            <Typography variant="subtitle1" gutterBottom>Parameter Randomization</Typography>
                                <FormControlLabel
                                control={<Checkbox defaultChecked />}
                                label="Randomize strategy parameter values for each run"
                            />
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>
                                Maximum randomization range (+/- %) from the original parameter value
                            </Typography>
                            <Slider
                                defaultValue={20}
                                aria-label="Parameter Randomization Percentage"
                                valueLabelDisplay="auto"
                                step={5}
                                marks
                                min={0}
                                max={100}
                            />
                        </Box>
                    </Box>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4, justifyContent: 'center', textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                            After running the simulations, you'll receive a summary report highlighting key statistics such as average return, maximum drawdown, and the percentage of profitable runs. This will help you gauge the strategy's robustness under varied conditions.
                        </Typography>
                    </Box>
                </Box>
                )}

                {mode === 'data_segmentation' && (
                    <Box sx={{ p: 3, display: 'flex', flexDirection: 'row', gap: 4, height: '100%' }}>
                        {/* Left Panel: Settings */}
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4, borderRight: 1, pr: 4, borderColor: 'divider', flex: 1 }}>
                            <Box>
                                <Typography variant="h6" sx={{ textAlign: 'center' }}>Data Segmentation</Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                                    Validate the strategy by optimizing on one period of historical data (In-Sample) and testing its performance on a completely separate, unseen period (Out-of-Sample). This is a crucial step to check for overfitting.
                                </Typography>
                            </Box>

                            {/* Window Configuration */}
                            <Box>
                                <Typography variant="subtitle1" gutterBottom sx={{ pb:2 }}>Window Configuration</Typography>
                                <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                        Adjust the sliders to partition your historical data into training, validation, and testing sets.
                                    </Typography>
                                    <Box sx={{ px: 2 }}>
                                        <Slider
                                            value={[trainingPct, trainingPct + validationPct]}
                                            onChange={(event, newValue) => {
                                                if (Array.isArray(newValue)) {
                                                    const newTrainingPct = newValue[0];
                                                    const newValidationEnd = newValue[1];
                                                    const newValidationPct = newValidationEnd - newTrainingPct;
                                                    
                                                    setTrainingPct(newTrainingPct);
                                                    setValidationPct(newValidationPct);
                                                    setTestingPct(100 - newValidationEnd);
                                                }
                                            }}
                                            valueLabelDisplay="off"
                                            disableSwap
                                            sx={{
                                                '& .MuiSlider-rail': {
                                                    backgroundImage: `linear-gradient(to right, #4caf50 ${trainingPct}%, #ff9800 ${trainingPct}%, #ff9800 ${trainingPct + validationPct}%, #f44336 ${trainingPct + validationPct}%)`,
                                                    height: 8,
                                                    borderRadius: 4,
                                                },
                                                '& .MuiSlider-track': {
                                                    display: 'none',
                                                },
                                            }}
                                        />
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                                        <Box sx={{ textAlign: 'center', color: '#4caf50' }}>
                                            <Typography variant="caption">Training</Typography>
                                            <Typography variant="h6">{trainingPct}%</Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'center', color: '#ff9800' }}>
                                            <Typography variant="caption">Validation</Typography>
                                            <Typography variant="h6">{validationPct}%</Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'center', color: '#f44336' }}>
                                            <Typography variant="caption">Testing</Typography>
                                            <Typography variant="h6">{testingPct}%</Typography>
                                        </Box>
                                    </Box>
                                </Box>
                            </Box>

                            <Box>
                                <TextField
                                    select
                                    fullWidth
                                    label="Optimization Metric"
                                    value={optimizationMetric}
                                    onChange={e => setOptimizationMetric(e.target.value)}
                                    variant="outlined"
                                    sx={{mt: 2}}
                                >
                                    <MenuItem value="Net_Profit">Net Profit</MenuItem>
                                    <MenuItem value="Avg_Monthly_Return">Avg Monthly Returns</MenuItem>
                                    <MenuItem value="Total_Trades">Total Trades</MenuItem>
                                    <MenuItem value="Max_Drawdown">Max Drawdown</MenuItem>
                                    <MenuItem value="Max_Drawdown_Duration">Max Drawdown Duration</MenuItem>
                                    <MenuItem value="Sharpe_Ratio">Sharpe Ratio</MenuItem>
                                    <MenuItem value="Profit_Factor">Profit Factor</MenuItem>
                                    <MenuItem value="Calmar_Ratio">Calmar Ratio</MenuItem>
                                    <MenuItem value="Equity_Efficiency_Rate">Equity Efficiency Rate</MenuItem>
                                    <MenuItem value="Strategy_Quality">Strategy Quality</MenuItem>
                                    <MenuItem value="Winrate">Win Rate</MenuItem>
                                </TextField>
                            </Box>

                            <Box>
                                <TextField
                                    fullWidth
                                    type="number"
                                    label="Number of Top Parameter Sets to pick for Validation & Testing"
                                    value={topParamSets} // <-- Set value
                                    onChange={e => setTopParamSets(parseInt(e.target.value, 20) || 0)} // <-- Add onChange
                                    variant="outlined"
                                    InputProps={{ inputProps: { min: 1, max: 1000 } }}
                                />
                            </Box>
                        </Box>
                        
                        {/* Right Panel: Explanation */}
                        <Box sx={{ display: 'flex', flexDirection: 'row', gap: 4, justifyContent: 'center', textAlign: 'center', flex: 2 }}>
                            <CommonPanels {...commonPanelProps} />
                        </Box>
                    </Box>
                )}

                {mode === 'walk_forward' && (
                    <Box sx={{ p: 3, display: 'flex', flexDirection: 'row', gap: 2, height: '100%' }}>
                        {/* Left Panel: Settings */}
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4, borderRight: 1, pr: 2, borderColor: 'divider', flex: 1 }}>
                            <Box>
                                <Typography variant="h6">Walk-Forward Analysis</Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Simulate real-world performance by optimizing on past data and testing on unseen future data, incrementally.
                                </Typography>
                            </Box>

                            {/* Window Configuration */}
                            <Box>
                                <Typography variant="subtitle1" gutterBottom sx={{ pb:2 }}>Window Configuration</Typography>
                                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                                    <TextField 
                                        fullWidth 
                                        type="number" 
                                        label="Training Period" 
                                        value={trainingPeriodLength} 
                                        onChange={e => setTrainingPeriodLength(parseInt(e.target.value, 10) || 0)} 
                                        variant="outlined" 
                                        InputProps={{ inputProps: { min: 1 } }} 
                                        />
                                    <Select value={trainingPeriodUnit} onChange={e => setTrainingPeriodUnit(e.target.value as any)} variant="outlined" sx={{ minWidth: 120 }}>
                                        <MenuItem value="days">Days</MenuItem>
                                        <MenuItem value="weeks">Weeks</MenuItem>
                                        <MenuItem value="months">Months</MenuItem>
                                    </Select>
                                </Box>
                                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                                    <TextField 
                                        fullWidth 
                                        type="number" 
                                        label="Testing Period" 
                                        value={testingPeriodLength} 
                                        onChange={e => setTestingPeriodLength(parseInt(e.target.value, 10) || 0)} 
                                        variant="outlined" 
                                        InputProps={{ inputProps: { min: 1 } }} 
                                        />
                                    <Select value={testingPeriodUnit} onChange={e => setTestingPeriodUnit(e.target.value as any)} variant="outlined" sx={{ minWidth: 120 }}>
                                        <MenuItem value="days">Days</MenuItem>
                                        <MenuItem value="weeks">Weeks</MenuItem>
                                        <MenuItem value="months">Months</MenuItem>
                                    </Select>
                                </Box>
                                <FormControlLabel 
                                    control={<Checkbox checked={isAnchored} 
                                    onChange={e => setIsAnchored(e.target.checked)} />} 
                                    label="Anchored Walk-Forward" 
                                    />
                                <Typography variant="caption" color="text.secondary" display="block">
                                    If checked, the training window start date is fixed and only the end date expands. Otherwise, it's a rolling window.
                                </Typography>
                            </Box>

                            {/* Step & Optimization */}
                            <Box>
                                <Typography variant="subtitle1" gutterBottom>Step & Optimization</Typography>
                                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>
                                    Step Forward Size (% of Testing Period)
                                </Typography>
                                <Slider 
                                    value={stepForwardPct} 
                                    onChange={(e, val) => setStepForwardPct(val as number)} 
                                    aria-label="Step Forward Size" 
                                    valueLabelDisplay="auto" 
                                    step={5} 
                                    marks 
                                    min={5} 
                                    max={100} 
                                    />
                                <TextField 
                                    select 
                                    fullWidth 
                                    label="Optimization Metric" 
                                    value={optimizationMetric} 
                                    onChange={e => setOptimizationMetric(e.target.value)} 
                                    variant="outlined" 
                                    sx={{ mt: 2 }}
                                    >
                                    <MenuItem value="Net_Profit">Net Profit</MenuItem>
                                    <MenuItem value="Avg_Monthly_Return">Avg Monthly Returns</MenuItem>
                                    <MenuItem value="Total_Trades">Total Trades</MenuItem>
                                    <MenuItem value="Max_Drawdown">Max Drawdown</MenuItem>
                                    <MenuItem value="Max_Drawdown_Duration">Max Drawdown Duration</MenuItem>
                                    <MenuItem value="Sharpe_Ratio">Sharpe Ratio</MenuItem>
                                    <MenuItem value="Profit_Factor">Profit Factor</MenuItem>
                                    <MenuItem value="Calmar_Ratio">Calmar Ratio</MenuItem>
                                    <MenuItem value="Equity_Efficiency_Rate">Equity Efficiency Rate</MenuItem>
                                    <MenuItem value="Strategy_Quality">Strategy Quality</MenuItem>
                                    <MenuItem value="Winrate">Win Rate</MenuItem>
                                </TextField>
                            </Box>
                        </Box>
                        {/* Right Panel: Explanation */}
                        <Box sx={{ display: 'flex', flexDirection: 'row', gap: 2, justifyContent: 'center', textAlign: 'center', flex: 2 }}>
                            <CommonPanels {...commonPanelProps} />
                        </Box>
                    </Box>
                )}

                {mode === 'correlated_assets' && (
                <Box sx={{ p: 3 }}>
                    <Typography variant="h6">Correlated Assets Test</Typography>
                    <Typography>Configuration for Correlated Assets Test will go here.</Typography>
                </Box>
                )}
                {mode === 'market_regime' && (
                <Box sx={{ p: 3 }}>
                    <Typography variant="h6">Market Regime Analysis</Typography>
                    <Typography>Configuration for Market Regime Analysis will go here.</Typography>
                </Box>
                )}
            </Box>
            )}
        </DialogContent>
        
        <DialogActions>
            <Button onClick={onClose} disabled={isSubmitting}>Cancel</Button>
            <Button onClick={handleSubmit} variant="contained" disabled={isLoading || isSubmitting}>Run Durability Test</Button>
        </DialogActions>

    </Dialog>
    );
};

const CombinationRuleRow: React.FC<{rule: CombinationRule, availableParams: OptimizableParameter[], onUpdate: Function, onRemove: Function}> = ({rule, availableParams, onUpdate, onRemove}) => {
    const operators = ['<', '>', '<=', '>=', '===', '!=='];
    return (
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 1, p: 1.5, borderRadius: 2, bgcolor: 'action.hover' }}>
            <Select size="small" value={rule.param1} onChange={e => onUpdate(rule.id, 'param1', e.target.value)} displayEmpty fullWidth>
                <MenuItem value="" disabled>Select Param...</MenuItem>
                {availableParams.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
            </Select>
            <Select size="small" value={rule.operator} onChange={e => onUpdate(rule.id, 'operator', e.target.value)}>
                {operators.map(op => <MenuItem key={op} value={op}>{op}</MenuItem>)}
            </Select>
            <Select size="small" value={rule.param2} onChange={e => onUpdate(rule.id, 'param2', e.target.value)} displayEmpty fullWidth>
                <MenuItem value="" disabled>Select Param...</MenuItem>
                {availableParams.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
            </Select>
            <IconButton onClick={() => onRemove(rule.id)} color="error"><DeleteOutlineIcon /></IconButton>
        </Box>
    );
}

// --- Helper Component for each parameter's input row (from previous step) ---
const ParameterInputRow: React.FC<{param: OptimizableParameter, handleParamChange: Function, isSubmitting: boolean}> = ({param, handleParamChange, isSubmitting}) => {
    return (
        <Box sx={{ p: 2, mb: 1, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <FormControlLabel
                    control={<Checkbox checked={param.enabled} onChange={e => handleParamChange(param.id, 'enabled', e.target.checked)} />}
                    label={<Typography sx={{ fontWeight: 500 }}>{param.name}</Typography>}
                />
                <ToggleButtonGroup
                    size="small"
                    value={param.mode}
                    exclusive
                    disabled={!param.enabled || isSubmitting}
                    onChange={(e, newMode) => { if (newMode) handleParamChange(param.id, 'mode', newMode) }}
                >
                    <ToggleButton value="range">Range</ToggleButton>
                    <ToggleButton value="list">List</ToggleButton>
                </ToggleButtonGroup>
            </Box>
            
            {param.mode === 'range' ? (
                <Box sx={{ display: 'flex', gap: 2, pl: 6 }}>
                    <TextField fullWidth label="Start" type="number" variant="outlined" size="small" value={param.start} onChange={e => handleParamChange(param.id, 'start', parseFloat(e.target.value) || 0)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                    <TextField fullWidth label="End" type="number" variant="outlined" size="small" value={param.end} onChange={e => handleParamChange(param.id, 'end', parseFloat(e.target.value) || 0)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                    <TextField fullWidth label="Step" type="number" variant="outlined" size="small" value={param.step} onChange={e => handleParamChange(param.id, 'step', parseFloat(e.target.value) || 1)} disabled={!param.enabled || isSubmitting} InputLabelProps={{ shrink: true }}/>
                </Box>
            ) : (
                <Box sx={{ pl: 6 }}>
                    <TextField
                        fullWidth
                        label="Values (comma-separated)"
                        variant="outlined"
                        size="small"
                        value={param.list_values}
                        onChange={e => handleParamChange(param.id, 'list_values', e.target.value)}
                        placeholder="e.g., 20, 50, 100, 155"
                        disabled={!param.enabled || isSubmitting}
                        InputLabelProps={{ shrink: true }}
                    />
                </Box>
            )}
        </Box>
    );
}