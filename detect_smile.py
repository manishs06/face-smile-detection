from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path of face cascade")
ap.add_argument("-m", "--model", required=True, help="path of trained model")
ap.add_argument("-v", "--video", help="path of video file")
ap.add_argument("-s", "--skip", type=int, default=2, help="number of frames to skip")
args = vars(ap.parse_args())

# Initialize face detector and model
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# Set video source
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

# Set camera properties for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

# Initialize variables
frame_count = 0
prev_time = time.time()
fps = 0

while True:
    (grabbed, frame) = camera.read()
    
    if args.get("video") and not grabbed:
        break
    
    # Skip frames for better performance
    frame_count += 1
    if frame_count % args["skip"] != 0:
        continue
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Resize frame for faster processing
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    
    # Optimize face detection parameters
    rects = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(150, 150),  # Add maximum size to reduce processing
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Process faces in batch if possible
    if len(rects) > 0:
        rois = []
        for (fX, fY, fW, fH) in rects:
            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float")/255.0
            rois.append(roi)
        
        # Batch prediction
        rois = np.array(rois)
        predictions = model.predict(rois, batch_size=32)
        
        # Draw results
        for i, (fX, fY, fW, fH) in enumerate(rects):
            (notSmiling, smiling) = predictions[i]
            label = "Smiling" if smiling > notSmiling else "Not Smiling"
            
            # Add confidence score
            confidence = max(smiling, notSmiling) * 100
            text = f"{label}: {confidence:.1f}%"
            
            cv2.putText(frameClone, text, (fX-20, fY-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
    
    # Display FPS
    cv2.putText(frameClone, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Face", frameClone)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
