from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize a YOLO-World model
    model = YOLO('yolov8s-world.pt')  # or choose yolov8m/l-world.pt

    # Define custom classes
    model.set_classes(["person",'car','license plate'])

    # Execute prediction for specified categories on an image
    results = model.predict('img_2.png',conf=0.5)

    # Show results
    results[0].show()