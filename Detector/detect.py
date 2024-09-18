from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO('yolov8n.pt')
results = model.predict(
   source=r"C:\Users\halabala\Documents\GitHub\WebElementDetector\screenshot.png",
   conf=0.25
)

img = cv2.imread(r"C:\Users\halabala\Documents\GitHub\WebElementDetector\screenshot.png")

for r in results:
    
    annotator = Annotator(img)
    
    boxes = r.boxes
    for box in boxes:
        
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
        
img = annotator.result()  
cv2.imshow('YOLO V8 Detection', img)     
cv2.waitKey(0)
cv2.destroyAllWindows()