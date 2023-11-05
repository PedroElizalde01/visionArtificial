import cv2 as cv
import joblib
from labels import int_to_label
import huMomentsGenerator as hg

WEBCAM_ID = 1
PREDICTOR_PATH = "data/knn_super_model.joblib"

def initialize_webcam(window_name):
    webcam = cv.VideoCapture(WEBCAM_ID)

    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(window_name, 1200, 674)

    denoise_trackbar_name = 'Denoise Level'
    cv.createTrackbar(denoise_trackbar_name, window_name, 1, 7, lambda a: None)

    binary_trackbar_name = 'Binary Threshold'
    cv.createTrackbar(binary_trackbar_name, window_name, 0, 255, lambda a: None)

    contour_size_trackbar_name = 'Contour Size'
    cv.createTrackbar(contour_size_trackbar_name, window_name, 2500, 50000, lambda a: None)

    return webcam, window_name, denoise_trackbar_name, binary_trackbar_name, contour_size_trackbar_name

def process_frame(webcam, window_name, denoise_trackbar, binary_trackbar, contour_size_trackbar):
    binary_value = cv.getTrackbarPos(binary_trackbar, window_name)
    radius = cv.getTrackbarPos(denoise_trackbar, window_name)

    _, original_image = webcam.read()
    original_image = cv.flip(original_image, 1)

    binary_image = hg.get_binary_image(original_image, binary_value)
    denoised_image = hg.denoise_image(binary_image, radius)
    cv.imshow(window_name, denoised_image)

    contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    second_largest_contour = hg.get_second_largest_contour(contours)
    return original_image, second_largest_contour

def main():
    window_name = "Image Processing"
    webcam, _, denoise_trackbar_name, binary_trackbar_name, contour_size_trackbar_name = initialize_webcam(window_name)

    knn_super_model = joblib.load(PREDICTOR_PATH)

    while True:
        original_image, second_largest_contour = process_frame(webcam, window_name, denoise_trackbar_name, binary_trackbar_name, contour_size_trackbar_name)

        if second_largest_contour is not None:
            cv.drawContours(original_image, [second_largest_contour], -1, (255, 0, 255), 3)
            cv.imshow(window_name, original_image)

            key = cv.waitKey(10)

            if key == ord('k'):
                hu_moments = hg.calculate_hu_moments(second_largest_contour)
                predicted_label = knn_super_model.predict([hu_moments])[0]
                predicted_label_str = int_to_label(predicted_label)
                print(f"Predicted Label: {predicted_label} ({predicted_label_str})")
                labeled_image = original_image.copy()
                cv.putText(labeled_image, f"Predicted Label: {predicted_label} ({predicted_label_str})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow("Labeled Contour", labeled_image)

            elif key == ord("x"):
                break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
