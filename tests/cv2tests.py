import cv2
from ultralytics import YOLO

#model = YOLO('yolov8n.pt')
model = YOLO('parking_lot/best.pt')


image_path = 'one-car.jpg'

results = model.predict(source=image_path, save=True, save_txt=True)

for r in results:
    processed_image = r.plot()

    cv2.imshow("detections", processed_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
