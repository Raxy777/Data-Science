import cv2
import numpy as np

cap = cv2.VideoCapture("noisy_video.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("denoised_video.mp4", fourcc, fps, (width, height))

# Buffer to store frames for temporal denoising
frame_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray_frame)

    # Keep only the last 5 frames in buffer
    if len(frame_buffer) > 5:
        frame_buffer.pop(0)

    # Apply Non-Local Means denoising when buffer has enough frames
    if len(frame_buffer) == 5:
        denoised_frame = cv2.fastNlMeansDenoisingMulti(
            srcImgs=frame_buffer,
            imgToDenoiseIndex=2,  
            temporalWindowSize=5,
            h=10  
        )

        denoised_bgr = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
        out.write(denoised_bgr)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Denoised Frame", denoised_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
