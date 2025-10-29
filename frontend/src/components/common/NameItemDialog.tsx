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

// Define the "contract" for the component's props
interface NameInputDialogProps {
  // Controls whether the dialog is visible
  open: boolean;
  
  // Function to call when the dialog should close (e.g., user clicks "Cancel" or outside the dialog)
  onClose: () => void;
  
  // Function to call when the user confirms the action. It receives the final input text.
  onConfirm: (name: string) => void;
  
  // --- Optional props for customization ---
  dialogTitle?: string;
  dialogText?: string;
  confirmButtonText?: string;
  initialValue?: string;
}

export const NameInputDialog: React.FC<NameInputDialogProps> = ({
  open,
  onClose,
  onConfirm,
  dialogTitle = "Enter Name", // Default title
  dialogText = "Please provide a name for the item.", // Default descriptive text
  confirmButtonText = "Confirm", // Default confirm button text
  initialValue = "", // Default initial value for the text field
}) => {
  // Internal state to manage the text field's value
  const [inputValue, setInputValue] = useState(initialValue);

  // Effect to reset the input value whenever the dialog is opened.
  // This is crucial for ensuring the correct initialValue is displayed each time.
  useEffect(() => {
    if (open) {
      setInputValue(initialValue);
    }
  }, [open, initialValue]);

  // Handler for the confirm button click
  const handleConfirm = () => {
    // Only proceed if the input is not just empty spaces
    if (inputValue.trim()) {
      onConfirm(inputValue.trim()); // Pass the trimmed value back to the parent
      onClose(); // Close the dialog after confirmation
    }
  };

  // Handler for pressing the Enter key in the text field
  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleConfirm();
    }
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>{dialogTitle}</DialogTitle>
      <DialogContent>
        <DialogContentText sx={{ mb: 2 }}>
          {dialogText}
        </DialogContentText>
        <TextField
          autoFocus // Automatically focus the input field when the dialog opens
          margin="dense"
          id="name"
          label="Name"
          type="text"
          fullWidth
          variant="outlined" // Using outlined for a more modern look
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="inherit">Cancel</Button>
        <Button 
          onClick={handleConfirm}
          variant="contained"
          // Disable the button if the input is empty to prevent invalid submissions
          disabled={!inputValue.trim()}
        >
          {confirmButtonText}
        </Button>
      </DialogActions>
    </Dialog>
  );
};