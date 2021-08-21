# ID_Greet

ID Greet is a real time AI facial recognition program. 

Author: Erik Scovin ######@###.### V1.8.01.2021
Microsoft Visual Studio Community 2019
Version 16.10.4
OpenCV

ID Greet uses a live webcam feed (or whatever camera is set as default) and OpenCV to identify faces in real time. The system is pre-trained with Haars Cascade Algo to discern faces from the enviornment. To load your own images for recognition, create an "Images" folder in the FaceDetection directory and within that folder create a folder named after the person to be identified (FaceDetection/Images/John Smith) and fill it with pictures of that person. Replace the any of the names in the greeting section of the code with the name of the person/folder and the system will trigger a greeting when that face is matched within the feed. Change the greeting by uploading a different .wav file. Live tracking with an outline box and display name will also occur when a face is recognized within the feed. 
