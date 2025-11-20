from ultralytics import YOLO
import time
import cv2

class parkingSpace:
    def __init__(self, id, coords, occupied):
        self.id = id
        self.coords = coords
        self.occupied = occupied
        
    def checkSpaceOccupancy(self, occupiedSpaces):
        for takenSpace in occupiedSpaces:
            if self.id == takenSpace.id:
                self.occupied = True
                print(self.id, "is occupied")
            else:
                self.occupied = False
                print(self.id, "not occupied")


# Load YOLO model
model = YOLO("best.pt")   # custom trained model for parking lot detection
#model.predict(source=4, show=False, conf=0.5) # use webcam as source
videoCap = cv2.VideoCapture(0)
time.sleep(1)
print(videoCap.isOpened())
videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
videoCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

isCalibrated = False
parkingSpaces = []
occupiedSpaces = []
interval = 3 # seconds between frame captures
lastCapturetime = 0

print("Starting camera...")

def spaceBeingOccupied(x, y, s): #Will return the ID of the space being occupied if there is a car present
    x1, y1, x2, y2 = s.coords
    if x1 < x < x2 and y1 < y < y2:
        return s.id
    else:
        return None

def groupParkingLinesByRow(parkingLines, yThreshold=300):
    row1 = []
    row2 = []
    allRows = []

    #create 2 seperate rows based on top rows bottom y and the bottom rows top y
    #sortedParkingLines = sorted(parkingLines,key=lambda r: r[1])

    for pLine in parkingLines:
        if pLine[1] < yThreshold and pLine[3] <yThreshold:
            row1.append(pLine)
        else:
            row2.append(pLine)

    allRows.append(row1)
    allRows.append(row2)
    return allRows

    
    # rows = []
    # for i in sorted(parkingLines,key=lambda r: r[1]): #go through parking lines by their y values to create rows
    #     inExistingRow = False #Bool to check if line belongs to existing row
    #     for row in rows:
    #         if abs(row[0][1] - i[1]) < yThreshold: #If the y coordinate is within threshold of existing row
    #             row.append(i) #Add this line to that row
    #             inExistingRow = True
    #             break
    #     if not inExistingRow:
    #         rows.append([i]) #Create new row with this line if still not in an existing row row
    # return rows

def calibrateParkingLines(frame):
    global isCalibrated
    global parkingSpaces
    #Calibration is done by pulling detected objects, checking if parking lines, then capturing their locations
    parkingLines = []
    #detectedObjects = model.track(frame, stream=True)
    detectedObjects = model.predict(frame, stream=True, imgsz=(800,600), conf=0.5)

    #Go through each detected object and retrieve appropriate info needed
    for objectInFrame in detectedObjects:
        
        #Only look at boxes (which surround objects with high confidence
        for box in objectInFrame.boxes:
            boxType = box.cls
            confidence = float(box.conf[0])
            if confidence > 0.5:
                if boxType == 1:
                    #x1,y1,x2,y2 = box.xyxy #returns bounding box coordinates
                    xyList = box.xyxy.tolist()
                    x1,y1,x2,y2 = int(xyList[0][0]),int(xyList[0][1]),int(xyList[0][2]),int(xyList[0][3])

                    parkingLines.append((x1,y1,x2,y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) #Draw rectangle around parking line MAY NEED TO REMOVE
                    cv2.putText(frame, "Parking Line", 
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2) #places label above parking line

    #Now that we have all parking lines, we can define parking spaces by their rows
    if len(parkingLines) >= 2: #if there are at least 2 parking lines detected
        allRows = groupParkingLinesByRow(parkingLines) #Group parking lines into rows based on y coordinate proximity
        idCount = 0
        for row in allRows: #go through each row of parking lines
            row.sort(key=lambda r: r[0]) #Sort parking lines in row by x coordinate
            for i in range(len(row) - 1):
                for x in range(len(row[i])):
                    firstX1, firstY1, firstX2, firstY2 = row[i] #Return leftmost parking line coordinates
                    secondX1, secondY1, secondX2, secondY2 = row[i+1] #Return next parking line coordinates
                    if row[i][x] < 300: #check which row it is in in order to calculate correct space coords
                        spaceX1 = min(firstX1, firstX2)
                        spaceX2 = max(secondX1, secondX2)
                        spaceY1 = max(firstY1,firstY2)
                        spaceY2 = min(secondY1, secondY2)
                    else:
                        spaceX1 = min(firstX1, firstX2)
                        spaceX2 = max(secondX1, secondX2)
                        spaceY1 = max(firstY1,firstY2)
                        spaceY2 = min(secondY1, secondY2)

                spaceCoords = (spaceX1,spaceY1, spaceX2, spaceY2) #Define parking space coordinates based on two parking lines
                parkingSpaces.append(parkingSpace(id=idCount, coords=spaceCoords, occupied = False)) #Create parking space object and add to list with unique ID
                idCount += 1 #increase ID count for next parking space
        isCalibrated = True
        print("Calibration complete.", len(parkingSpaces), "parking spaces defined.")
    else:
        print("Not enough parking lines detected for calibration.")

def drawParkingSpaces(frame):
    global parkingSpaces

    # Draw spaces first
    for s in parkingSpaces:
        x1, y1, x2, y2 = s.coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "Space {}".format(s.coords),
                    (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
 
def detectCars(frame):
    global parkingSpaces
    global occupiedSpaces
    detectedObjects = model.predict(frame, stream=True)
    
    #boxType = box.cls
    #confidence = float(box.conf[0])

    for objectInFrame in detectedObjects:
        for box in objectInFrame.boxes:
            boxType = box.cls
            confidence = float(box.conf[0])
            if boxType == 0:
                if confidence > 0.4:
                    cls = int(boxType[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Draw the detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.putText(frame, class_name,
                    #             (x1, max(y1 - 10, 20)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                # Check if the toy car is inside a space
                # if boxType == 0:
                for s in parkingSpaces:
                    space_id = spaceBeingOccupied(cx, cy, s)
                    if space_id:
                        occupiedSpaces.append(space_id)

# print("Frame width = ", int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)))
# print("Frame height = ", int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
while True:
    currentTime = time.time()

    if currentTime - lastCapturetime >= interval:
        lastCapturetime = currentTime
        ret, frame = videoCap.read()

        if not ret:
            print("No frame captured. Exiting.")
            break

        if not isCalibrated:
            calibrateParkingLines(frame)
        else:
            drawParkingSpaces(frame)
            detectCars(frame)
            for space in parkingSpaces:
                space.checkSpaceOccupancy(occupiedSpaces)
           # print("Woo, calibrated")
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
            break
        
        

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
