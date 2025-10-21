// src/components/pipeline/SidePanel/colorUtils.tsx

/**
 * Generates a color for the heatmap using a diverging red-white-green scheme.
 * - Strong Negative (-1): Red
 * - Neutral (0): White/Light Gray
 * - Strong Positive (+1): Green
 */
export const getColorForCorrelation = (value: number): string => {
    // Ensure value is between -1 and 1
    const clampedValue = Math.max(-1, Math.min(1, value));

    // Define our three main colors
    const negativeColor = { r: 211, g: 71, b: 71 };    // A clear red
    const neutralColor = { r: 215, g: 215, b: 215 };  // Pure white
    const positiveColor = { r: 97, g: 186, b: 97 };   // A clear green

    if (clampedValue > 0) {
        // Positive correlation: Interpolate from neutral (white) to positive (green)
        const r = Math.round(neutralColor.r * (1 - clampedValue) + positiveColor.r * clampedValue);
        const g = Math.round(neutralColor.g * (1 - clampedValue) + positiveColor.g * clampedValue);
        const b = Math.round(neutralColor.b * (1 - clampedValue) + positiveColor.b * clampedValue);
        return `rgb(${r}, ${g}, ${b})`;

    } else if (clampedValue < 0) {
        // Negative correlation: Interpolate from neutral (white) to negative (red)
        const factor = Math.abs(clampedValue);
        const r = Math.round(neutralColor.r * (1 - factor) + negativeColor.r * factor);
        const g = Math.round(neutralColor.g * (1 - factor) + negativeColor.g * factor);
        const b = Math.round(neutralColor.b * (1 - factor) + negativeColor.b * factor);
        return `rgb(${r}, ${g}, ${b})`;
        
    } else {
        // Exactly zero correlation
        return `rgb(${neutralColor.r}, ${neutralColor.g}, ${neutralColor.b})`;
    }
};