from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("object-detection/yolo26n.pt") 

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, conf=0.5, stream=True)

    # Visualize results on the frame
    for r in results:
        frame = r.plot()

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()