from ultralytics import YOLO
import time
import cv2

class parkingSpace:
    def __init__(self, id, coords):
        self.id = id
        self.coords = coords


# Load YOLO model
model = YOLO("best.pt")   # custom trained model for parking lot detection
model.predict(source=0, show=False) # use webcam as source
videoCap = cv2.VideoCapture(0)

isCalibrated = False
parkingSpaces = []
interval = 3.0  # seconds between frame captures
lastCapturetime = 0

print("Starting camera...")

def spaceBeingOccupied(x, y, parkingSpaces): #Will return the ID of the space being occupied if there is a car present
    for s in parkingSpaces:
        x1, y1, x2, y2 = s["coords"]
        if x1 < x < x2 and y1 < y < y2:
            return s["id"]
    return None

def groupParkingLinesByRow(parkingLines, yThreshold=50):
    rows = []
    for i in sorted(parkingLines,key=lambda r: r[1]): #go through parking lines by their y values to create rows
        inExistingRow = False #Bool to check if line belongs to existing row
        for row in rows:
            if abs(row[0][1] - i[1]) < yThreshold: #If the y coordinate is within threshold of existing row
                row.append(i) #Add this line to that row
                inExistingRow = True
                break
        if not inExistingRow:
            rows.append([i]) #Create new row with this line if still not in an existing row row
    return rows

def calibrateParkingLines(frame):
    #Calibration is done by pulling detected objects, checking if parking lines, then capturing their locations
    parkingLines = []
    detectedObjects = model.track(frame, stream=True)

    #Go through each detected object and retrieve appropriate info needed
    for objectInFrame in detectedObjects:
        objectName = objectInFrame.names
        #Only look at boxes (which surround objects with high confidence
        for box in objectInFrame.boxes:
            confidence = float(box.conf[0])
            if confidence > 0.5:
                if objectName == "parking line":
                    x1,y1,x2,y2 = box.xyxy #returns bounding box coordinates
                    parkingLines.append((x1,y1,x2,y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) #Draw rectangle around parking line MAY NEED TO REMOVE
                    cv2.putText(frame, "Parking Line", 
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2) #places label above parking line

    #Now that we have all parking lines, we can define parking spaces by their rows
    if len(parkingLines) >= 2: #if there are at least 2 parking lines detected
        rows = groupParkingLinesByRow(parkingLines) #Group parking lines into rows based on y coordinate proximity
        idCount = 0
        for row in rows: #go through each row of parking lines
            row.sort(key=lambda r: r[0]) #Sort parking lines in row by x coordinate
        for i in range(len(row) - 1):
            firstX1, firstY1, firstX2, firstY2 = parkingLines[i] #Return leftmost parking line coordinates
            secondX1, secondY1, secondX2, secondY2 = parkingLines[i+1] #Return next parking line coordinates
            spaceCoords = (firstX2, min(firstY1, secondY1), secondX1, max(firstY2, secondY2)) #Define parking space coordinates based on two parking lines
            parkingSpaces.append(parkingSpace(id=idCount, coords=spaceCoords)) #Create parking space object and add to list with unique ID
            idCount += 1 #increase ID count for next parking space
        isCalibrated = True
        print("Calibration complete.", len(parkingSpaces), "parking spaces defined.")
    else:
        print("Not enough parking lines detected for calibration.")

def DrawParkingSpaces(frame, parkingSpaces):
    #results = model.track(frame, stream=True)
    occupied_spaces = set()

    # Draw spaces first
    for s in parkingSpaces:
        x1, y1, x2, y2 = s["coords"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "Space {}".format(s["id"]),
                    (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    

while True:
    currentTime = time.time()

    if currentTime - lastCapturetime >= interval:
        lastCapturetime = currentTime
        ret, frame = videoCap.read()

        if not ret:
            print("No frame captured. Exiting.")
            break

        if not isCalibrated:
            parkingSpaces = calibrateParkingLines(frame)
        else:
            DrawParkingSpaces(frame, parkingSpaces)

cv2.destroyAllWindows()

# while True:
#     #Capture frame every time interval
#     currentTime = time.time()

#     if currentTime - lastCapturetime >= interval:
#         #update time, and read frame
#         lastCapturetime = currentTime
#         ret, frame = videoCap.read()

#         #exit if no frame captured
#         if not ret:
#             print("No frame captured. Exiting.")
#             break

#     #Check if calibration is needed on this frame
#         #Calbrate

#     if isCalibrated == False:
#         calibrateParkingLines(frame)
#     else:
#         DrawParkingSpaces(frame, parkingSpaces)

    #else detect if car is occupying any existing spaces.
