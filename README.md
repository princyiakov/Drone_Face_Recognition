# Drone_Face_Recognition
## README yet to be updated as new features like front end for missing person registration, Integration with database has been updated recently . Stay Tuned ! 
## How does Face Recognition work ?
1.The face detection process is an essential step as it detects and locates human faces in images and videos.
2.The face capture process transforms analog information (a face) into a set of digital information (data) based on the person's facial features.
3. The face match process verifies if two faces belong to the same person.

## Let us Understand the Face Recognition better 
Here is a interesting read for you  : https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/ 

## Understanding FaceNet and MTCNN
https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/

## What does this project do? 

Provides a drone which can be navigated by user and can recognize individuals who have been reported missing (data updated in a database and photos available for training the model)

## How do we achieve it ? 

 1. Reads data from an existing database containing the record of missing people. (Used a csv record here)

 2. Facial Recognition : Google FaceNet's MTCNN model extracts the face from the images  and Inception ResnetV1 provides the embeddings of the face and is stored along with the data retrieved from database in ".pt" file, implemented on Pytorch and compared with the faces transmitted live by drone.

 3. Navigation of drone : Pygame is used to navigate the drone.

## How to run the code ?
1. `git clone https://github.com/princys-laboratory/Drone_Face_Recognition.git`
2. `pip install -r requirements_pip.txt`
3. `mkdir Images_FaceNet`
4. For adding images of person : add folders and follow ID (0,1,...) : `mkdir 0`
5. Add Images of each person in each folder 
6. Change the database.csv data according to the data added
7. To train the model with the added data : `python training.py`
8. 'known_faces.pt' will be created
3. To run the Face Recognition using Tello : `python run.py`
