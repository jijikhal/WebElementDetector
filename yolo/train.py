# https://docs.ultralytics.com/models/yolov8/#performance-metrics

from ultralytics import YOLO

if __name__ == "__main__":

    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\yolo\dataset\dataset.yaml", epochs=100, imgsz=640)

    # Run inference with the YOLOv8n model on the 'bus.jpg' image
    results = model("path/to/bus.jpg")