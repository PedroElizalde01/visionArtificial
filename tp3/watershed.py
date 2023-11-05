import cv2
import numpy as np

WEBCAM_ID = 1

def initialize_variables():
    base_colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0], [100, 255, 0],
                    [100, 0, 255], [0, 100, 255], [0, 255, 100], [255, 100, 0]]
    frame_window = 'Frame-Window'
    seeds_map_window = 'Seeds-Map-Window'
    watershed_result_window = 'Watershed-Result-Window'
    return base_colours, frame_window, seeds_map_window, watershed_result_window

def initialize_webcam(webcam_id):
    cap = cv2.VideoCapture(webcam_id)
    _, frame = cap.read()
    h, w, _ = frame.shape
    seeds = np.zeros((h, w), np.uint8)
    return cap, frame, seeds, (h, w)

def click_event(event, x, y, _flags, _params):
    if event == cv2.EVENT_LBUTTONDOWN:
        val = int(chr(selected_key))
        points.append(((x, y), val))
        cv2.circle(seeds, (x, y), 7, (val, val, val), thickness=-1)
    return points, seeds

def watershed(img, seeds, base_colours, watershed_result_window):
    markers = cv2.watershed(img, np.int32(seeds))
    img[markers == -1] = [0, 0, 255]
    for n in range(1, 10):
        img[markers == n] = base_colours[n]
    cv2.imshow(watershed_result_window, img)
    cv2.waitKey()

def main():
    base_colours, frame_window, seeds_map_window, watershed_result_window = initialize_variables()

    cap, frame, seeds, sizeTuple = initialize_webcam(WEBCAM_ID)

    selected_key = 49  # 1 en ASCII
    points = []

    while True:
        _, frame = cap.read()
        frame_copy = frame.copy()
        seeds_copy = seeds.copy()

        for point in points:
            color = point[1]
            val = point[1] * 20
            x = point[0][0]
            y = point[0][1]
            cv2.circle(frame_copy, (x, y), 7, val, thickness=-1)
            cv2.circle(seeds_copy, (x, y), 7, val, thickness=-1)
            cv2.putText(frame_copy, str(point[1]), (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 3)

        cv2.imshow(frame_window, frame_copy)
        map = cv2.applyColorMap(seeds_copy, cv2.COLORMAP_JET)
        cv2.imshow(seeds_map_window, map)

        key = cv2.waitKey(100) & 0xFF
        if key == 32:
            watershed(frame.copy(), seeds, base_colours, watershed_result_window)
            points = []
            seeds = np.zeros(sizeTuple, np.uint8)

        if ord('1') <= key <= ord('9'):
            selected_key = key

        if key == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
        main()
