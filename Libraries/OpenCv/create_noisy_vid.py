import cv2 as cv
import numpy as np

capture = cv.VideoCapture('Resources/Videos/dog.mp4')

fps = int(capture.get(cv.CAP_PROP_FPS))
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')  
out = cv.VideoWriter("noisy_video.mp4", fourcc, fps, (width, height))

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break 

    row, col, ch = frame.shape

    mean = 0
    var = 50 
    sigma = var ** 0.5

    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.int16)

    noisy_frame = np.clip(frame.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

    out.write(noisy_frame)

    cv.imshow('Noisy Video', noisy_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
out.release()
cv.destroyAllWindows()
