import cv2
import time

cap = cv2.VideoCapture(0)


# Function: displayCamera
# Params: None
# Return: Camera Status True = Camera is operating
#                       False = Camera is not operating/was terminated
# Description: This function will capture a frame every 3 seconds and
#              will continue to do so until the user ends operation by
#              pressing 'q'

def displayCamera():
    
    time.sleep(3) #Only capture image every 3 seconds to save computing later on
    ret, frame = cap.read() # Read a frame from the camera

    if not ret: # Check if the frame was read successfully
        print("Error: Can't receive frame (stream end?). Exiting ...")
        return False

    cv2.imshow('Camera Feed', frame) # Display the captured frame in a window

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    
    return True


def main():

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
       displayStatus = displayCamera()
       if displayStatus == False:
           break


main()
cap.release()
cv2.destroyAllWindows()
print("Operation Terminated")
