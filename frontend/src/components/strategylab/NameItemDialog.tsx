import React, { useState, useEffect } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
} from '@mui/material';

interface NameInputDialogProps {
  open: boolean;
  // Let's make the title and button text configurable
  dialogTitle: string;
  dialogText: string;
  confirmButtonText: string;
  initialValue?: string; // For pre-filling the rename field
  onClose: () => void;
  onConfirm: (name: string) => void;
}



export const NameInputDialog: React.FC<NameInputDialogProps> = ({
    open,
    dialogTitle,
    dialogText,
    confirmButtonText,
    initialValue = '',
    onClose,
    onConfirm,
  }) => {
  const [name, setName] = useState(initialValue);

  // Set a default name and clear input when the dialog opens
  useEffect(() => {
    if (open) {
      setName(initialValue);
    }
  }, [initialValue, open]);

  const handleConfirm = () => {
    if (name.trim()) {
      onConfirm(name.trim());
      onClose();
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleConfirm();
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>{dialogTitle}</DialogTitle>
      <DialogContent>
        <DialogContentText>{dialogText}</DialogContentText>
        <TextField
          autoFocus
          margin="dense"
          id="name"
          label="Name"
          type="text"
          fullWidth
          variant="standard"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyPress={handleKeyPress}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleConfirm} variant="contained">{confirmButtonText}</Button>
      </DialogActions>
    </Dialog>
  );
};