#! ./hex_map_extract/bin/python3
import cv2

import cv2
import numpy as np
import math

def get_average_color_in_circular_region(image, center_x, center_y, radius):
    """
    Extracts the average color from a circular region in an image and
    returns its brightly saturated BGR version.

    Args:
        image (numpy.ndarray): The input image (BGR).
        center_x (int): The x-coordinate of the circle's center.
        center_y (int): The y-coordinate of the circle's center.
        radius (int): The radius of the circle.

    Returns:
        tuple: A tuple (B, G, R) representing the brightly saturated average
               color, or None if the circle is entirely outside the image,
               the ROI is invalid, or contains no pixels.
    """
    height, width = image.shape[:2]

    # Ensure center and radius are integers for drawing and calculations
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    radius = int(round(radius))

    # Quick check if the circle's bounding box is completely outside the image
    if (center_x + radius <= 0 or center_x - radius >= width or
            center_y + radius <= 0 or center_y - radius >= height):
        # print("Debug: Circle's bounding box is completely outside image.")
        return None

    # Define the bounding box for the circular region (Region of Interest - ROI)
    x_start = max(0, center_x - radius)
    y_start = max(0, center_y - radius)
    x_end = min(width, center_x + radius)
    y_end = min(height, center_y + radius)

    # If the ROI has no area, it means the circle (or its relevant part) is outside
    if x_start >= x_end or y_start >= y_end:
        # print(f"Debug: ROI is invalid or outside. x_start:{x_start}, x_end:{x_end}, y_start:{y_start}, y_end:{y_end}")
        return None

    # Create a mask for the circular region, but only for the ROI
    # Adjust center coordinates relative to the ROI
    roi_center_x = center_x - x_start
    roi_center_y = center_y - y_start

    # Create mask for the ROI
    mask_roi = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)
    cv2.circle(mask_roi, (roi_center_x, roi_center_y), radius, 255, -1)

    # Ensure the mask within the ROI has some non-zero pixels
    if cv2.countNonZero(mask_roi) == 0:
        # print("Debug: No non-zero pixels in mask_roi.")
        return None

    # Extract the ROI from the image
    image_roi = image[y_start:y_end, x_start:x_end]

    # Calculate the average color using the image ROI and mask ROI
    # This addresses the TODO: cv2.mean now operates on a smaller image region
    average_color_bgr_roi = cv2.mean(image_roi, mask=mask_roi)

    # cv2.mean returns (B, G, R, Alpha/0), we only need BGR
    bgr_tuple = tuple(map(int, average_color_bgr_roi[:3]))

    # Convert BGR tuple to a 1x1x3 NumPy array for cv2.cvtColor
    bgr_np_pixel = np.uint8([[bgr_tuple]])

    # Convert BGR to HSV
    hsv_np_pixel = cv2.cvtColor(bgr_np_pixel, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_np_pixel[0, 0]

    # Check if the color is reasonably close to black
    black_value_threshold = 20
    if v < black_value_threshold:
        return (0, 0, 0)  # Output pure black

    # Check if the color is reasonably close to white
    white_value_threshold = 205
    white_saturation_threshold = 20
    if v > white_value_threshold and s < white_saturation_threshold:
        return (255, 255, 255)  # Output pure white

    # Create a mutable copy for modification (or access directly if careful)
    # hsv_np_pixel is [[[h, s, v]]]
    # To make it brightly saturated: set S to 255 and V to 255
    hsv_bright_pixel = hsv_np_pixel.copy()
    hsv_bright_pixel[0, 0, 1] = 255  # Saturation
    hsv_bright_pixel[0, 0, 2] = 255  # Value

    # Convert the brightly saturated HSV color back to BGR
    ret_bgr_np_pixel = cv2.cvtColor(hsv_bright_pixel, cv2.COLOR_HSV2BGR)

    # Extract the BGR tuple from the 1x1x3 NumPy array
    ret_bgr_tuple = tuple(map(int, ret_bgr_np_pixel[0, 0]))

    return ret_bgr_tuple
def extract_hex_grid_hues(image_path, start_x, start_y, grid_rows, grid_cols, dot_radius, horizontal_spacing,
                          vertical_spacing, visualize):
    """
        Extracts average colors from circular regions in a hexagonal grid on an image.

        Args:
            image_path (str): Path to the input image.
            start_x (int): X-coordinate of the initial point (center of the first dot in the top-left).
            start_y (int): Y-coordinate of the initial point.
            grid_rows (int): Number of rows in the hexagonal grid.
            grid_cols (int): Number of columns in the hexagonal grid.
            dot_radius (int): Radius of the circular regions (dots).
            horizontal_spacing (float): Horizontal distance between centers of dots in the same row.
            vertical_spacing (float): Vertical distance between rows of dots.
            visualize (bool): If True, displays the image with circles and their average colors.

        Returns:
            list: A list of dictionaries, where each dictionary contains:
                  'center': (cx, cy),
                  'avg_color_bgr': (B, G, R)
                  Returns an empty list if the image cannot be loaded.
        """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []

    output_image = image.copy() if visualize else None
    average_colors_data = []
    image_height, image_width = image.shape[:2]

    raw_colors = []

    print(f"Image dimensions: Width={image_width}, Height={image_height}")
    print(f"Grid: {grid_rows} rows, {grid_cols} columns")
    print(f"Dot radius: {dot_radius}")
    print(f"Spacing: Horizontal={horizontal_spacing}, Vertical={vertical_spacing}")
    print(f"Starting point: ({start_x}, {start_y})")

    for r in range(grid_rows):
        raw_colors.append([])

        for c in range(grid_cols):
            center_x = start_x + c * horizontal_spacing
            center_y = start_y + r * vertical_spacing

            # Offset for hexagonal pattern (odd rows)
            if r % 2 == 1:
                center_x += horizontal_spacing / 2.0

            # Ensure coordinates are integers for processing
            cx_int, cy_int = int(round(center_x)), int(round(center_y))

            # Basic check if the center is way outside (can be refined)
            if not (0 <= cx_int < image_width and 0 <= cy_int < image_height):
                print(f"Skipping point ({cx_int}, {cy_int}) as it's outside typical image center range.")
                # Continue if you want to attempt circles near edges,
                # get_average_color_in_circular_region will handle exact boundary checks.
                # If you want to strictly skip centers outside image bounds: continue

            avg_color = get_average_color_in_circular_region(image, cx_int, cy_int, dot_radius)
            # avg_color = (255, 255, 255)

            if avg_color:
                raw_colors[-1].append('#{:02x}{:02x}{:02x}'.format(avg_color[2], avg_color[1], avg_color[0]))
                average_colors_data.append({
                    'center': (cx_int, cy_int),
                    'avg_color_bgr': avg_color
                })
                if visualize and output_image is not None:
                    # Draw the circle outline
                    cv2.circle(output_image, (cx_int, cy_int), int(dot_radius), avg_color,
                               2)  # Use average color for outline
                    cv2.circle(output_image, (cx_int, cy_int), 1, (0, 0, 255), -1)  # Mark center
                    # Optionally, put text for the color
                    # color_text = f"({avg_color[0]},{avg_color[1]},{avg_color[2]})"
                    # cv2.putText(output_image, color_text, (cx_int + dot_radius + 2, cy_int),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)
            else:
                print(
                    f"Could not get average color for dot at ({cx_int}, {cy_int}) with radius {dot_radius}. It might be outside image bounds or too small.")

    if visualize and output_image is not None:
        print(f'colors: \n\n{raw_colors}')
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', output_image)
        cv2.resizeWindow('custom window', 600, 600)

        # cv2.imshow("Hexagonal Grid Average Colors", output_image)
        print("\nPress any key to close the visualization window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return average_colors_data


if __name__ == "__main__":

    initial_x = 63.0
    initial_y = 39.0

    horiz_span_A = 3703.0 - 41.0
    cols_A = (158.0 - 2.0) / 2.0 + 1.0

    vert_last = 2842.0 + 12.0
    vert_span = vert_last - initial_y
    rows = 107.0 * 2.0 - 2

    cols = cols_A

    region_radius = 5.0

    horiz_spacing = horiz_span_A / cols_A
    vert_spacing = vert_span / rows

    # cv2.imshow("Image", image)
    # cv2.waitKey(0) # Wait for a key press
    # cv2.destroyAllWindows()


    extracted_hues = extract_hex_grid_hues(
        image_path = 'map.jpg',
        start_x = initial_x,
        start_y = initial_y,
        grid_rows = int(rows),
        grid_cols = int(cols),
        dot_radius = region_radius,
        horizontal_spacing = horiz_spacing,
        vertical_spacing = vert_spacing,
        visualize = True
    )
