if (typeof window.dash_clientside === 'undefined') {
    window.dash_clientside = {};
}
if (typeof window.dash_clientside.clientside === 'undefined') {
    window.dash_clientside.clientside = {};
}

/**
 * Implements a NumPy-like percentile function in JavaScript.
 * By default, q should be in [0, 100].
 *
 * @param {number[]} arr - 1D array of numbers
 * @param {number} q - Percentile in [0, 100]
 * @returns {number} The percentile value
 */
function np_percentile(arr, q) {
    if (!arr.length) {
        return NaN;
    }
    const sorted = [...arr].sort((a, b) => a - b);

    // Edge cases
    if (q <= 0) {
        return sorted[0];
    }
    if (q >= 100) {
        return sorted[sorted.length - 1];
    }

    // Linear interpolation
    const n = sorted.length;
    const index = (n - 1) * (q / 100);
    const i = Math.floor(index);
    const frac = index - i;
    if (i + 1 >= n) {
        return sorted[i];
    }
    return sorted[i] * (1 - frac) + sorted[i + 1] * frac;
}

/**
 * Dash Clientside Callback
 * Normalizes raw numerical data to [0–255] using percentile cutoffs (like your Python code),
 * optionally applying an image-based mask (alpha channel).
 *
 * If no mask is provided, negative and NaN values are masked out, matching your Python code.
 *
 * @param {boolean} logToggle     - (Currently unused but preserved for your signature)
 * @param {Array}   percentileRange - [lowPerc, highPerc], e.g. [0.01, 99]
 * @param {string}  maskSrc         - (optional) URL for mask image
 * @param {Array}   rawData         - 1D or 2D numeric array
 *
 * @returns {string} A base64 PNG data URL
 */
window.dash_clientside.clientside.transform_image = function(logToggle, percentileRange, maskSrc, rawData) {
    console.log("Received logToggle:", logToggle);
    console.log("Received percentileRange:", percentileRange);
    console.log("Received maskSrc:", maskSrc);
    console.log("Received rawData:", typeof rawData, Array.isArray(rawData) ? rawData.length : "not array");

    // If there's no raw data or percentile range is invalid, return an empty base64 string
    if (!rawData || !Array.isArray(percentileRange) || percentileRange.length !== 2) {
        console.log("No valid raw data or percentile range. Returning empty base64 image.");
        // Return an empty data URL for a transparent pixel
        return Promise.resolve("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=");
    }

    // Extract low/high percentile from the range slider
    var lowPerc = percentileRange[0];
    var highPerc = percentileRange[1];

    // If there's a mask, we need to load it
    var maskImage = new Image();
    var maskPromise;

    if (maskSrc) {
        maskPromise = new Promise(function(resolve) {
            maskImage.onload = resolve;
            maskImage.onerror = function(err) {
                console.warn("Could not load mask. Proceeding without a mask.");
                resolve(null);
            };
            maskImage.src = maskSrc;
        });
    } else {
        maskPromise = Promise.resolve(null);
    }

    // Process data once the mask is loaded (if any)
    return maskPromise.then(function() {
        // Work with raw data directly
        // Assume rawData is a flat array of values or a 2D array that needs to be flattened
        var flatRawData = Array.isArray(rawData[0]) ? rawData.flat() : rawData;
        var width = Array.isArray(rawData[0]) ? rawData[0].length : Math.sqrt(flatRawData.length);
        var height = Array.isArray(rawData[0]) ? rawData.length : Math.sqrt(flatRawData.length);

        // Make sure width and height are integers
        width = Math.floor(width);
        height = Math.floor(height);

        console.log("Processing raw data of dimensions:", width, "x", height);

        // Get mask data if available
        var maskArray = null;
        if (maskSrc && maskImage.width > 0 && maskImage.height > 0) {
            var maskCanvas = document.createElement("canvas");
            var maskCtx = maskCanvas.getContext("2d");
            // Resize the mask canvas to match the rawData dimensions
            maskCanvas.width = width;
            maskCanvas.height = height;

            maskCtx.drawImage(maskImage, 0, 0, maskCanvas.width, maskCanvas.height);
            var maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

            maskArray = new Uint8ClampedArray(maskData.length / 4);
            for (var i = 0, j = 0; i < maskData.length; i += 4, j++) {
                var alpha = maskData[i + 3];
                maskArray[j] = alpha > 128 ? 1 : 0;
            }
        }

        // 1) Collect valid intensities for percentile computation
        var intensities = [];
        var threshold = 1e-12; // For log transform
        var hasMask = (maskArray !== null);

        for (var i = 0; i < flatRawData.length; i++) {
            var val = flatRawData[i];

            // Skip masked or invalid values
            if (hasMask && maskArray[i] === 0) {
                continue;
            }

            // Skip negative and NaN values if we're doing log transform
            if (logToggle && (isNaN(val) || val < 0)) {
                continue;
            }

            // For non-log transform, include negative values but skip NaN
            if (!logToggle && isNaN(val)) {
                continue;
            }

            // Apply log transform if needed
            if (logToggle) {
                val = Math.max(threshold, val);
                val = Math.log(val + threshold);
            }

            intensities.push(val);
        }

        // If no valid intensities found, return an empty data URL
        if (intensities.length === 0) {
            console.log("No valid intensities found. Returning empty image.");
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        }

        // 2) Calculate percentiles
        intensities.sort(function(a, b) { return a - b; });

        // Helper function to get percentile value
        function getPercentileValue(sortedArray, percentile) {
            if (sortedArray.length === 0) return 0;

            var index = Math.ceil((percentile / 100) * sortedArray.length) - 1;
            index = Math.max(0, Math.min(sortedArray.length - 1, index));

            return sortedArray[index];
        }

        var lowVal = getPercentileValue(intensities, lowPerc);
        var highVal = getPercentileValue(intensities, highPerc);

        console.log("Intensity range after preprocessing:", lowVal, "to", highVal);

        // Make sure we have a valid range to prevent division by zero
        if (lowVal === highVal) {
            highVal = lowVal + 1;
        }

        // 3) Transform raw data to create final image
        var imageDataArray = new Uint8ClampedArray(width * height * 4); // RGBA

        for (var i = 0; i < Math.min(flatRawData.length, width * height); i++) {
            var val = flatRawData[i];
            var pixelIndex = i * 4;

            // Handle invalid values
            if (isNaN(val)) {
                imageDataArray[pixelIndex] = 0;     // R
                imageDataArray[pixelIndex + 1] = 0; // G
                imageDataArray[pixelIndex + 2] = 0; // B
                imageDataArray[pixelIndex + 3] = 255; // A (fully opaque)
                continue;
            }

            // Handle masked values
            if (hasMask && maskArray[i] === 0) {
                imageDataArray[pixelIndex] = 0;     // R
                imageDataArray[pixelIndex + 1] = 0; // G
                imageDataArray[pixelIndex + 2] = 0; // B
                imageDataArray[pixelIndex + 3] = 0; // A (transparent)
                continue;
            }

            // Apply log transform if needed
            if (logToggle && val > 0) {
                val = Math.max(threshold, val);
                val = Math.log(val + threshold);
            }

            // Clip to percentile range
            val = Math.max(lowVal, Math.min(highVal, val));

            // Normalize to [0,1] range
            var normalized = (val - lowVal) / (highVal - lowVal);

            // Scale to 0-255 range
            var finalVal = Math.round(normalized * 255);

            // Set pixel values (grayscale)
            imageDataArray[pixelIndex] = finalVal;     // R
            imageDataArray[pixelIndex + 1] = finalVal; // G
            imageDataArray[pixelIndex + 2] = finalVal; // B
            imageDataArray[pixelIndex + 3] = 255;      // A (fully opaque)
        }

        // Create a canvas and put the image data
        var canvas = document.createElement("canvas");
        var ctx = canvas.getContext("2d");
        canvas.width = width;
        canvas.height = height;

        try {
            // Create ImageData object and put on canvas
            var imageDataObj = new ImageData(imageDataArray, width);
            ctx.putImageData(imageDataObj, 0, 0);

            // Return the full data URL including the prefix
            return canvas.toDataURL();
        } catch (err) {
            console.error("Error creating image:", err);
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        }
    })
    .catch(function(err) {
        console.error("Failed to transform image:", err);
        // Return an empty data URL
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    });
};
