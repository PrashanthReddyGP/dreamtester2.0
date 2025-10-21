// src/components/strategylab/CsvImportDialog.tsx
import React, { useState, useRef } from 'react';
import {
    Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField,
    MenuItem, Box, Typography, CircularProgress, Alert
} from '@mui/material';

const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];
const sources = ['MetaTrader', 'Kaggle', 'Other'];

interface CsvImportDialogProps {
    open: boolean;
    onClose: () => void;
    onSubmit: (symbol: string, timeframe: string, source: string, file: File) => Promise<void>;
}

export const CsvImportDialog: React.FC<CsvImportDialogProps> = ({ open, onClose, onSubmit }) => {
    const [symbol, setSymbol] = useState('');
    const [timeframe, setTimeframe] = useState('1h');
    const [source, setSource] = useState('MetaTrader');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
        setSelectedFile(event.target.files[0]);
        }
    };
    
    const resetState = () => {
        setSymbol('');
        setTimeframe('1h');
        setSource('MetaTrader');
        setSelectedFile(null);
        setError(null);
        setIsSubmitting(false);
        // Also reset the file input so the same file can be selected again
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleClose = () => {
        resetState();
        onClose();
    };

    const handleSubmit = async () => {
        if (!symbol || !timeframe || !source || !selectedFile) {
        setError('All fields are required.');
        return;
        }
        setError(null);
        setIsSubmitting(true);
        try {
        await onSubmit(symbol.toUpperCase(), timeframe, source, selectedFile);
        handleClose(); // Close dialog on success
        } catch (err: any) {
        setError(err.message || 'An unexpected error occurred.');
        } finally {
        setIsSubmitting(false);
        }
    };
    
    const isFormInvalid = !symbol || !timeframe || !source || !selectedFile || isSubmitting;

    return (
        <Dialog open={open} onClose={handleClose} fullWidth maxWidth="xs">
        <DialogTitle>Import OHLCV Data from CSV (MetaTrader)</DialogTitle>
        <DialogContent>
            <TextField
            autoFocus
            margin="dense"
            label="Asset Symbol (e.g., EURUSD)"
            type="text"
            fullWidth
            variant="outlined"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            sx={{ mt: 2 }}
            />
            <TextField
            select
            margin="dense"
            label="Timeframe"
            fullWidth
            variant="outlined"
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            >
            {timeframes.map((option) => (
                <MenuItem key={option} value={option}>
                {option}
                </MenuItem>
            ))}
            </TextField>
            <TextField
                select
                margin="dense"
                label="Source"
                fullWidth
                variant="outlined"
                value={source}
                onChange={(e) => setSource(e.target.value)}
            >
                {sources.map((option) => (
                    <MenuItem key={option} value={option}>
                        {option}
                    </MenuItem>
                ))}
            </TextField>
            <Box mt={2}>
            <Button variant="outlined" component="label" fullWidth>
                Choose CSV File
                <input type="file" hidden accept=".csv" ref={fileInputRef} onChange={handleFileChange} />
            </Button>
            {selectedFile && (
                <Typography variant="body2" color="text.secondary" mt={1}>
                Selected: {selectedFile.name}
                </Typography>
            )}
            </Box>
            {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        </DialogContent>
        <DialogActions sx={{ p: '0 24px 16px' }}>
            <Button onClick={handleClose} disabled={isSubmitting}>Cancel</Button>
            <Button onClick={handleSubmit} variant="contained" disabled={isFormInvalid}>
            {isSubmitting ? <CircularProgress size={24} /> : 'Import Data'}
            </Button>
        </DialogActions>
        </Dialog>
    );
};