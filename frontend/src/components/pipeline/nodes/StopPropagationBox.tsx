// src/components/common/StopPropagationBox.tsx
import React from 'react';

// This component wraps any children and stops keyboard events from bubbling up.
export const StopPropagationBox: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => {
    const handleKeyDown = (e: React.KeyboardEvent) => {
        // Stop the event from propagating to the global keydown listener in PipelineEditor
        e.stopPropagation();
    };

    return (
        <div onKeyDown={handleKeyDown} className={className}>
        {children}
        </div>
    );
};