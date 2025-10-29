// src/components/common/SaveTemplateDialog.tsx

import React, { useState, useEffect } from 'react';
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    TextField,
    Stack
} from '@mui/material';

interface SaveTemplateDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (name: string, description: string) => void;
  title: string;
}

export const SaveTemplateDialog: React.FC<SaveTemplateDialogProps> = ({ open, onClose, onSave, title }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');

  // This effect runs whenever the `open` prop changes.
  useEffect(() => {
    // We only reset the state when the dialog is opened.
    if (open) {
      setName('');
      setDescription('');
      setError('');
    }
  }, [open]); // The dependency array ensures this runs only when `open` changes.

  const handleSave = () => {
    if (!name.trim()) {
      setError('Template name is required.');
      return;
    }
    onSave(name, description);
    onClose(); // This will trigger the useEffect on the next open
  };

  const handleClose = () => {
    onClose(); // This will also trigger the useEffect on the next open
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
            <DialogContentText>
                Please provide a name and an optional description for your new template.
            </DialogContentText>
            <TextField
                autoFocus
                required
                margin="dense"
                id="name"
                label="Template Name"
                type="text"
                fullWidth
                variant="outlined"
                value={name}
                onChange={(e) => setName(e.target.value)}
                error={!!error}
                helperText={error}
            />
            <TextField
                margin="dense"
                id="description"
                label="Description (Optional)"
                type="text"
                fullWidth
                multiline
                rows={2}
                variant="outlined"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
            />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">Save</Button>
      </DialogActions>
    </Dialog>
  );
};