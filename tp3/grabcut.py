import cv2 as cv
import numpy as np

WEBCAM_ID = 1

def grabcut_and_display():
    # Abre la cámara web utilizando el ID de la cámara especificado (WEBCAM_ID)
    camera = cv.VideoCapture(WEBCAM_ID)

    # Espera a que el usuario presione ENTER para capturar una imagen
    print('Press ENTER to capture the image')
    input()

    # Captura una imagen de la cámara
    _, image = camera.read()
    # Voltea la imagen horizontalmente (espejo)
    image = cv.flip(image, 1)

    # Crea una máscara inicial de ceros del mismo tamaño que la imagen
    mask = np.zeros(image.shape[:2], np.uint8)

    # Permite al usuario seleccionar una región de interés (ROI) en la imagen
    rectangle = cv.selectROI("Select frame", image, fromCenter=False, showCrosshair=True)

    # Inicializa GrabCut con la máscara y la región rectangular seleccionada
    cv.grabCut(image, mask, rectangle, None, None, 10, cv.GC_INIT_WITH_RECT)

    # Crea una nueva máscara (mask2) donde los píxeles etiquetados como "Fondo" o "Desconocido" se establecen en 0 y los "Primer plano" en 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Aplica la máscara a la imagen original para aislar la región de interés
    image = image * mask2[:, :, np.newaxis]

    # Muestra la imagen resultante con la máscara aplicada
    cv.imshow("Output with the mask", image)
    cv.waitKey()

if __name__ == "__main__":
    grabcut_and_display()
    cv.destroyAllWindows()

