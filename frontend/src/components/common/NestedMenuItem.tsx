// src/components/shared/NestedMenuItem.tsx

import React, { useState, useRef } from 'react';
import { Menu, MenuItem } from '@mui/material';
import type { PopoverOrigin } from '@mui/material';
import type { MenuItemProps } from '@mui/material';

// Define the props our component will accept
export interface NestedMenuItemProps extends Omit<MenuItemProps, 'onClick'> {
    label: string;
    parentMenuOpen: boolean;
    onClick?: () => void;
    children?: React.ReactNode;
}

const NestedMenuItem = React.forwardRef<HTMLLIElement, NestedMenuItemProps>((props, ref) => {
    const {
        label,
        parentMenuOpen,
        children,
        ...menuItemProps
    } = props;

    const menuItemRef = useRef<HTMLLIElement>(null);
    const [isSubMenuOpen, setIsSubMenuOpen] = useState(false);

    const handleMouseEnter = () => {
        setIsSubMenuOpen(true);
    };

    const handleMouseLeave = () => {
        setIsSubMenuOpen(false);
    };

    // Determine if the main menu is open and we have something to show in a submenu
    const isSubmenuExist = !!children;

    const anchorOrigin: PopoverOrigin = { vertical: 'top', horizontal: 'right' };
    const transformOrigin: PopoverOrigin = { vertical: 'top', horizontal: 'left' };

    return (
        <MenuItem
            {...menuItemProps}
            ref={menuItemRef}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            sx={{
                display: 'flex',
                justifyContent: 'space-between',
                // Add an arrow to indicate a submenu
                '&:after': isSubmenuExist ? {
                    content: '"›"',
                    marginLeft: '8px',
                } : {},
            }}
        >
            {label}
            {isSubmenuExist && (
                <Menu
                    // Style the submenu
                    sx={{ pointerEvents: 'none' }}
                    anchorEl={menuItemRef.current}
                    anchorOrigin={anchorOrigin}
                    transformOrigin={transformOrigin}
                    open={isSubMenuOpen && parentMenuOpen}
                    // The onClose is handled by the parent, but we prevent pointer events on the Menu itself
                    onClose={() => { /* This is intentionally left blank */ }}
                >
                    {/* The magic: we wrap the children to re-enable pointer events */}
                    <div style={{ pointerEvents: 'auto' }}>
                        {children}
                    </div>
                </Menu>
            )}
        </MenuItem>
    );
});

export default NestedMenuItem;