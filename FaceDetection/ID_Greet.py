
from imutils import paths
import simpleaudio as sa
import face_recognition
import pickle
import cv2
import imutils
import time
import os
 
#Get paths of each file in folder named Images
imagePaths = list(paths.list_images('Images'))
kEncodings = []
kNames = []
count = 0
jing = 0
jang = 0
jip = 0
#Loop over image paths
for (i, imagePath) in enumerate(imagePaths):
    #Extracts name from file
    name = imagePath.split(os.path.sep) [-2]
    #Loads and converts input image
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    #Loop over encodings
    for encoding in encodings:
        kEncodings.append(encoding)
        kNames.append(name)

#Save encodings with name in dictionary
data = {"encodings": kEncodings, "names": kNames}
#Use pickle to save for later use
f = open("face_enc", "wb")
#Opens file
f.write(pickle.dumps(data))
#Closes file
f.close()

#Recognizes faces in webcam stream

#Loads pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Load known faces and embeddings saved pickle
data = pickle.loads(open('face_enc', "rb").read())
print("Streaming started")
video_capture = cv2.VideoCapture(0)
#Loop over frames from video file stream

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    #Convert frame imput from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    #Loop over facial embeddings in case of multiple faces
    for encoding in encodings:
        #Compare encodings with data encodings
        #Matches contrain array with boolean values and True for the embeddings closely matched, false for rest
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        #Check for match
        if True in matches:
            #Find position True is returned and store
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            #Loop over marched indexes and count recognized faces
            for i in matchedIdxs:
                #Check names at indexes
                name = data["names"][i]
                #Increase count for the name returned
                counts[name] = counts.get(name, 0) + 1
            #Set name with highest count
            name = max(counts, key=counts.get)

            #Update list of names
            names.append(name)
            #Loop over recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            #Rescale face coordinSates
            #Draw predicted face name on image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            #Greets recognized face with pre-recorded greeting.
            if 'Erik' == name and count == 0:
                count = count + 1
                filename = 'Erik.wav'
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            if 'Brianna' == name and jing == 0:
                jing = jing + 1
                filename = 'Brianna.wav'
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            if 'Scott' == name and jing == 0:
                jang = jang + 1
                filename = 'Scott.wav'
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            if 'Susan' == name and jing == 0:
                jip = jip + 1
                filename = 'Susan.wav'
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done()

    cv2.imshow("Face Recognition AI", frame)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
video_capture.release()
cv2.destroyAllWindows()