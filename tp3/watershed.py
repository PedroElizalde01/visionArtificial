import cv2
import numpy as np

# ID de la cámara web que se utilizará
WEBCAM_ID = 1

# Definición de colores base para etiquetar regiones en la imagen
base_colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0], [100, 255, 0], [100, 0, 255], [0, 100, 255], [0, 255, 100], [255, 100, 0]]

# Nombres de las ventanas de visualización
frame_window = 'Frame-Window'
seeds_map_window = 'Seeds-Map-Window'
watershed_result_window = 'Watershed-Result-Window'

# Función para aplicar Watershed a la imagen
def watershed(img):
    # Aplica el algoritmo Watershed para segmentar la imagen
    markers = cv2.watershed(img, np.int32(seeds))
    # Marca los píxeles que no se han etiquetado como fondo con color rojo
    img[markers == -1] = [0, 0, 255]

    # Asigna colores a las regiones segmentadas en base a las semillas
    for n in range(1, 10):
        img[markers == n] = base_colours[n]

    # Muestra la imagen segmentada en la ventana 'Watershed-Result-Window'
    cv2.imshow(watershed_result_window, img)
    cv2.waitKey()

# Función para manejar eventos de clic en la imagen
def click_event(event, x, y, _flags, _params):
    if event == cv2.EVENT_LBUTTONDOWN:
        val = int(chr(selected_key))
        points.append(((x, y), val))
        # Dibuja una semilla en la posición de clic con el color correspondiente
        cv2.circle(seeds, (x, y), 7, (val, val, val), thickness=-1)

# Función principal
def main():
    global points
    global seeds
    global frame
    global selected_key
    selected_key = 49  # Valor ASCII para '1'
    points = []

    # Inicializa la cámara web
    cap = cv2.VideoCapture(WEBCAM_ID)
    _, frame = cap.read()
    h, w, _ = frame.shape

    seeds = np.zeros((h, w), np.uint8)
    cv2.namedWindow(frame_window)
    cv2.namedWindow(seeds_map_window)

    # Configura el manejo de eventos de clic en la ventana de la imagen
    cv2.setMouseCallback(frame_window, click_event)

    # Bucle principal
    print('SPACE to watershed')
    print('(1 to 9) to change number')
    print('q to quit')
    while True:
        _, frame = cap.read()
        frame_copy = frame.copy()
        seeds_copy = seeds.copy()

        # Dibuja las semillas y etiquetas en la imagen
        for point in points:
            color = point[1]
            val = point[1] * 20
            x = point[0][0]
            y = point[0][1]
            cv2.circle(frame_copy, (x, y), 7, val, thickness=-1)
            cv2.circle(seeds_copy, (x, y), 7, val, thickness=-1)
            cv2.putText(frame_copy, str(point[1]), (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

        # Muestra la imagen original y el mapa de semillas
        cv2.imshow(frame_window, frame_copy)
        map = cv2.applyColorMap(seeds_copy, cv2.COLORMAP_JET)
        cv2.imshow(seeds_map_window, map)

        # Captura la tecla presionada
        key = cv2.waitKey(100) & 0xFF

        # Acciones basadas en la tecla presionada
        if key == 32:  # Barra espaciadora para aplicar Watershed
            watershed(frame.copy())
            points = []
            seeds = np.zeros(variables.sizeTuple, np.uint8)
        if ord('1') <= key <= ord('9'):  # Cambiar la etiqueta de intensidad
            selected_key = key
        if key == ord('q'):  # 'q' para salir del programa
            break

    # Libera la cámara web al finalizar
    cap.release()

if __name__ == '__main__':
    main()
