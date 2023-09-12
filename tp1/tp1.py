import cv2 as cv
import constants


# Function to create a window with optional width and height
def create_window(name, width=600, height=337):
    cv.namedWindow(name, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(name, width, height)

# Function to create a trackbar
def create_trackbar(window_name, trackbar_name, initial_value, max_value, callback):
    cv.createTrackbar(trackbar_name, window_name, initial_value, max_value, callback)

# Function to initialize the windows and trackbars
def initialize_windows_and_trackbars():
    # Denoise Window
    denoise_window_name = 'Binary Image'
    create_window(denoise_window_name)
    create_trackbar(denoise_window_name, 'Noise', 1, 7, lambda a: None)
    create_trackbar(denoise_window_name, 'Binary', 0, 255, lambda a: None)

    # Original Image Window
    original_image_window_name = 'Original Image'
    create_window(original_image_window_name)
    create_trackbar(original_image_window_name, 'Contour size', 2500, 10000, lambda a: None)

# Function to obtain a binary image from a given image using a threshold value
def get_binary_image(image, value):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_image, value, 255, cv.THRESH_BINARY)
    return thresh

# Function to denoise a binary image using morphological operations
def denoise_image(binary_image, radius):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

# Function to obtain contours from an image given a threshold value
def get_contours_by_image(image_route, thresh_bottom):
    shape = cv.imread(image_route)
    gray_shape = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)
    _, shape_thresh = cv.threshold(gray_shape, thresh_bottom, 255, cv.THRESH_BINARY_INV)
    shape_contours, _ = cv.findContours(shape_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if shape_contours:
        return shape_contours[0]
    else:
        return []  # Return an empty list if no contours are found

# Function to check if a contour matches a predefined shape's contour
def does_contour_match_shapes_contour(contour_shape, contour):
    if len(contour_shape) > 0 and len(contour) > 0:  # Check if both contours are non-empty
        return cv.matchShapes(contour_shape, contour, 1, 0.0) < 0.03
    else:
        return False  # If either contour is empty, consider them not matching

# Function to display an invalid shape on the original image
def display_invalid_shape(contour, original_image):
    x, y, _, _ = cv.boundingRect(contour)
    cv.putText(original_image, "Invalid", (x, y), cv.FONT_ITALIC, 1.5, (255, 255, 255), 1, cv.LINE_4)
    cv.drawContours(original_image, contour, -1, (0, 0, 255), 3)

# Function to display a valid shape on the original image
def display_valid_shape(contour, shape_name, original_image):
    x, y, _, _ = cv.boundingRect(contour)
    cv.putText(original_image, shape_name, (x, y), cv.FONT_ITALIC, 1.5, (255, 255, 255), 1, cv.LINE_4)
    cv.drawContours(original_image, contour, -1, (0, 255, 0), 3)

# Main function
def main():
    webcam = cv.VideoCapture(constants.WEBCAM_ID)
    initialize_windows_and_trackbars()
    key = 'a'

    # Define contour prototypes with corresponding names and thresholds
    contour_and_contour_names = [
        (get_contours_by_image(constants.SQUARE_PATH, 0), 'Square'),
        (get_contours_by_image(constants.TRIANGLE_PATH, 0), 'Triangle'),
        (get_contours_by_image(constants.CIRCLE_PATH, 0), 'Circle'),
        (get_contours_by_image(constants.STAR_PATH, 0), 'Star'),
    ]


    while key != ord('z'):
        # Get values from trackbars
        binary_value = cv.getTrackbarPos('Binary', 'Binary Image')
        radius = cv.getTrackbarPos('Noise', 'Binary Image')
        shape_contour_size = cv.getTrackbarPos('Contour size', 'Original Image')

        _, original_image = webcam.read()
        original_image = cv.flip(original_image, 1)

        binary_image = get_binary_image(original_image, binary_value)
        denoised_image = denoise_image(binary_image, radius)
        cv.imshow('Binary Image', denoised_image)

        contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for contour in contours:
            if cv.contourArea(contour) > shape_contour_size:
                all_defined_shapes_invalid = True
                for contour_shape, contour_name in contour_and_contour_names:
                    if does_contour_match_shapes_contour(contour_shape, contour):
                        all_defined_shapes_invalid = False
                        display_valid_shape(contour, contour_name, original_image)
                        break
                if all_defined_shapes_invalid:
                    display_invalid_shape(contour, original_image)

        cv.imshow('Original Image', original_image)
        key = cv.waitKey(1)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()