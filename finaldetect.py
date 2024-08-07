import cv2
import numpy as np
import pytesseract
# H847xC 172
plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
video = cv2.VideoCapture('Data/vid1.mp4')
if not video.isOpened():
    print('Error Reading Video')
target_plate_number = input("Enter the target license plate number: ")
detected = False
resize_factor = 0.5 
fps = video.get(cv2.CAP_PROP_FPS)
delay = int(500 / fps)  



while True:
    ret, frame = video.read()
    if not ret:
        break

    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plat_detector.detectMultiScale(gray_video, scaleFactor=1.5, minNeighbors=5, minSize=(25, 25))

    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]
        recognized_text = pytesseract.image_to_string(plate_img, config='--psm 8')

        print(f"Detected License Plate: {recognized_text}")

        # Set the color based on whether the target plate is detected
        if recognized_text.strip() == target_plate_number.strip():
            box_color = (0, 255, 0)  # Green color for detected plate
            detected = True
        else:
            box_color = (255, 0, 0)  # Red color for other plates

        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], ksize=(1, 1))
        cv2.putText(frame, recognized_text, org=(x-3, y-3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    color=(0, 0, 255), thickness=1, fontScale=0.6)

        if detected:
            print("Target license plate detected! Pausing the video.")
            break  # Break out of the loop to pause the video

    # Resize the frame to fit in the window
    height, width = frame.shape[:2]
    new_dim = (int(width * resize_factor), int(height * resize_factor))
    resized_frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

    # Save frame to debug
    cv2.imwrite('debug_frame.jpg', resized_frame)

    # Display frame
    cv2.imshow('Video', resized_frame)

    if detected:
        key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        if key == ord('q'):
            break
        elif key == ord('p'):
            detected = False  # Resume playback on 'p' key
    else:
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()





# P 001 AM 77




    #
    

