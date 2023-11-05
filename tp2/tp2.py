import cv2 as cv
import functools

WEBCAM_ID = 1

def create_and_resize_window(window_name, width, height):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(window_name, width, height)

def create_and_initialize_trackbar(window_name, trackbar_name, default_value, max_value):
    cv.createTrackbar(trackbar_name, window_name, default_value, max_value, lambda a: None)

def get_binary_image(image, value):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray_image, value, 255, cv.THRESH_BINARY)
    return binary_image

def denoise_image(binary_image, radius):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

def get_biggest_contour(contours):
    return functools.reduce(lambda a, b: a if cv.contourArea(a) > cv.contourArea(b) else b, contours)

def main():
    webcam = cv.VideoCapture(WEBCAM_ID)

    # Create and resize windows
    create_and_resize_window('Binary Image', 600, 337)
    create_and_resize_window('Original Image', 600, 337)

    # Create and initialize trackbars
    create_and_initialize_trackbar('Binary Image', 'Noise', 1, 7)
    create_and_initialize_trackbar('Binary Image', 'Binary', 0, 255)

    key = 'a'

    while key != ord('z'):
        binary_value = cv.getTrackbarPos('Binary', 'Binary Image')
        radius = cv.getTrackbarPos('Noise', 'Binary Image')

        _, original_image = webcam.read()
        original_image = cv.flip(original_image, 1)

        binary_image = get_binary_image(original_image, binary_value)

        denoised_image = denoise_image(binary_image, radius)
        cv.imshow('Binary Image', denoised_image)

        contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        biggest_contour = get_biggest_contour(contours)

        # Display or process the biggest contour as needed

        cv.imshow('Original Image', original_image)

        key = cv.waitKey(30)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
