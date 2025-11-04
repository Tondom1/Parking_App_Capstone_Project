from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolov8s.pt")   # or "yolo11s.pt" if using YOLOv11
videoCap = cv2.VideoCapture(0)

calibrated = False
spaces = []
interval = 5.0  # seconds between frame captures
last_capture_time = 0

print("Starting camera... Capturing frames every 5 seconds.")

def point_in_space(x, y, spaces):
    """Return the ID of the space containing a point, or None."""
    for s in spaces:
        x1, y1, x2, y2 = s["coords"]
        if x1 < x < x2 and y1 < y < y2:
            return s["id"]
    return None

while True:
    current_time = time.time()

    # Capture a frame every 5 seconds
    if current_time - last_capture_time >= interval:
        last_capture_time = current_time

        ret, frame = videoCap.read()
        if not ret:
            print("No frame captured. Exiting.")
            break

        # ---- STEP 1: Calibrate once ----
        if not calibrated:
            print("Running calibration (detecting remotes)...")
            results = model.track(frame, stream=True)
            remotes = []

            for result in results:
                class_names = result.names
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        cls = int(box.cls[0])
                        class_name = class_names[cls]
                        if "remote" in class_name.lower():
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            remotes.append((x1, y1, x2, y2))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "Remote",
                                        (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 2)

            if len(remotes) >= 2:
                remotes.sort(key=lambda r: r[0])
                for i in range(len(remotes) - 1):
                    left = remotes[i]
                    right = remotes[i + 1]

                    x1_space = left[2]
                    x2_space = right[0]
                    y1_space = min(left[1], right[1])
                    y2_space = max(left[3], right[3])

                    if x2_space > x1_space:
                        space = {
                            "id": i + 1,
                            "coords": (x1_space, y1_space, x2_space, y2_space)
                        }
                        spaces.append(space)

                if spaces:
                    calibrated = True
                    print("Calibration complete. Spaces detected:")
                    for s in spaces:
                        x1, y1, x2, y2 = s["coords"]
                        print("  Space {}: ({}, {}) -> ({}, {})".format(
                            s["id"], x1, y1, x2, y2))
                else:
                    print("Remotes found, but no valid spaces detected yet...")
            else:
                print("No remotes found during calibration.")

        # ---- STEP 2: After calibration, detect objects and check space occupancy ----
        else:
            results = model.track(frame, stream=True)
            occupied_spaces = set()

            # Draw spaces first
            for s in spaces:
                x1, y1, x2, y2 = s["coords"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, "Space {}".format(s["id"]),
                            (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

            # Check all detections for people
            for result in results:
                class_names = result.names
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        cls = int(box.cls[0])
                        class_name = class_names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Draw the detection
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, class_name,
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 0, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                        # Check if the cup is inside a space
                        if "cup" in class_name.lower():
                            space_id = point_in_space(cx, cy, spaces)
                            if space_id:
                                occupied_spaces.add(space_id)

            # Highlight occupied spaces
            for s in spaces:
                if s["id"] in occupied_spaces:
                    x1, y1, x2, y2 = s["coords"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "Occupied",
                                (x1 + 10, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    print("Person detected inside Space {}".format(s["id"]))

        # Show the processed snapshot
        cv2.imshow("YOLO Space Detection", frame)
        print("Frame processed. Waiting 5 seconds for next capture.")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

videoCap.release()
cv2.destroyAllWindows()