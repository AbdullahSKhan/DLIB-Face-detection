import face_recognition
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils 

video_capture = VideoStream(src=0).start()

ask_image = face_recognition.load_image_file("ask.jpg")
abdullah_face_encoding = face_recognition.face_encodings(ask_image)[0]

uzair_image = face_recognition.load_image_file("uzair.jpg")
uzair_face_encoding = face_recognition.face_encodings(uzair_image)[0]

#array me save krega or live train ho raha he
known_face_encodings = [
    abdullah_face_encoding,
    uzair_face_encoding
]
known_face_names = [
    "Abdullah",
    "Uzair"
]
fps = FPS().start()

while True:
    frame = video_capture.read()
    frame =imutils.resize(frame,width=500)
    
    #rgb_frame = frame[:, :, ::-1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
    face_locations = face_recognition.face_locations(gray)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

  
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):      
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
   
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Video', frame)

   
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()
fps.stop()
video_capture.release()
cv2.destroyAllWindows()
