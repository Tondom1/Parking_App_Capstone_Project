from ultralytics import YOLO
import time
import cv2

class parkingSpace:
    def __init__(self, id, coords):
        self.id = id
        self.coords = coords


# Load YOLO model
model = YOLO("/home/evan/school/capstone/train12-new/weights/best.pt")   # custom trained model for parking lot detection
#model.predict(source=4, show=False, conf=0.5) # use webcam as source
videoCap = cv2.VideoCapture(4)
time.sleep(1)
print(videoCap.isOpened())
videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
videoCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

isCalibrated = False
parkingSpaces = []
interval = 0 # seconds between frame captures
lastCapturetime = 0

print("Starting camera...")

def spaceBeingOccupied(x, y, parkingSpaces): #Will return the ID of the space being occupied if there is a car present
    for s in parkingSpaces:
        x1, y1, x2, y2 = s["coords"]
        if x1 < x < x2 and y1 < y < y2:
            return s["id"]
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
                parkingSpaces.append(parkingSpace(id=idCount, coords=spaceCoords)) #Create parking space object and add to list with unique ID
                idCount += 1 #increase ID count for next parking space
        isCalibrated = True
        print("Calibration complete.", len(parkingSpaces), "parking spaces defined.")
    else:
        print("Not enough parking lines detected for calibration.")

def DrawParkingSpaces(frame):
    global parkingSpaces
    #results = model.track(frame, stream=True)
    occupied_spaces = set()

    # Draw spaces first
    for s in parkingSpaces:
        x1, y1, x2, y2 = s.coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "Space {}".format(s.coords),
                    (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    
print("Frame width = ", int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Frame height = ", int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
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
            DrawParkingSpaces(frame)
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
