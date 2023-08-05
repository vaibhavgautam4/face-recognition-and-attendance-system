import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition

path = 'C:/Users/Vaibhav/FaceRecog/face/images'
pathnew = 'C:/Users/Vaibhav/FaceRecog/face/newimages'
video_path = cv2.VideoCapture(0)

# Define the known focal length of your camera (in millimeters)
focal_length = 800

# Define the size of an object of known distance (in millimeters)
object_width = 200

# Load the Haar Cascade classifier for detecting faces
# face_cascade = cv2.CascadeClassifier("extra_files/haarcascade_frontalface_default.xml")

# Load the Haar Cascade classifier for detecting full body
full_body_cascade = cv2.CascadeClassifier('extra_files/haarcascade_fullbody.xml')

images =[]
# images2 = []
classnames = []
# classnames2 = []
mylist = os.listdir(path)
mylist2 = os.listdir(pathnew)

print(mylist)
for cl in mylist:
  curimg = cv2.imread(f'{path}/{cl}')
  images.append(curimg)
  classnames.append(os.path.splitext(cl)[0])
print(classnames)

# print(mylist2)
# for cl2 in mylist2:
#   curimg2 = cv2.imread(f'{pathnew}/{cl2}')
#   images2.append(curimg2)
#   classnames2.append(os.path.splitext(cl2)[0])
# print(classnames2)

class video:
  def findEncodings(images):
    encodelist = []
    for img in images:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      encode = face_recognition.face_encodings(img)[0]
      encodelist.append(encode)
    return encodelist

  encodelistknown = findEncodings(images)
  print('encoding complete')

  def markattendance(name):
    with open('C:/Users/Vaibhav/FaceRecog/face/attendance.csv','r+') as f:
      mydatalist = f.readlines()
      namelist = []
      for line in mydatalist:
        entry = line.split(',')
        namelist.append(entry[0])
      if name not in namelist:
        now = datetime.now()
        dtstring = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtstring}')  

  def create_dir(path):
      try:
          if not os.path.exists(path):
              os.makedirs(path)
      except OSError:
          print(f"Error: creating directory with name {path}")
  
  def save_frame(video_path, path):
      name = 'vaibhav gautam'
      print(name)
      # save_path = os.path.join(path, name)
      video.create_dir(path)
      cv2.imwrite(f"{name}.jpg", video_path)

  # def add_photo(video):
  #     cv2.imshow('webcam' , img)
  #     cv2.imwrite('vaibhav_gautam_20_05-02-22_MayurVihar.jpg', pathnew)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # gray_frame = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the frame to grayscale
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize illumination in the grayscale frame
    # normalized_frame = cv2.equalizeHist(gray_frame)

    # Convert the normalized grayscale frame to RGB
    # rgb_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2RGB)


    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # facesCurFrame = face_recognition.face_locations(rgb_frame)
    # encodesCurFrame = face_recognition.face_encodings(rgb_frame, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(video.encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(video.encodelistknown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            # Calculate the size of the face in the image (in pixels)
            face_width_pixels = ( x2 - x1 )

            # Calculate the distance to the face based on the size of the face in the image
            distance = (object_width * focal_length) / face_width_pixels

            # Draw the distance on the frame
            cv2.putText(img, f"Face Distance: {distance:.2f} mm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            video.markattendance(name)

            # video.add_photo(cap)
            # from sklearn.metrics import accuracy_score
            # acc = accuracy_score(video.encodelistknown[] , encodesCurFrame)
            # print(acc)
        else:
          ret, frame = cap.read()
          facesCurrentFrame = face_recognition.face_locations(imgS)
          # facesCurrentFrame = face_recognition.face_locations(rgb_frame)
          video.save_frame(facesCurrentFrame, pathnew)
# how to add histogram equilization?


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1000) == 13:
        break

cap.release()
cv2.destroyAllWindows()

# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix
# data = np.load('olivetti_faces.npy')
# target = np.load('olivetti_faces_target.npy')
# convert array into dataframe
# data=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test=train_test_split(data, target, test_size=0.3, stratify=target, random_state=0)

# df = np.array_split(data,2)
# print(df2)
# print(data)
# print(target)
# DF = pd.DataFrame(df[0], index=np.arange(0,200))
# DF2 = pd.DataFrame(df[1], index=np.arange(0,200))

# DF.to_csv("data1.csv")
# DF2.to_csv("data2.csv")

# print(X_train)
# print(X_test)
# d1 = pd.read_csv('data1.csv')
# d2 = pd.read_csv('data2.csv')

# df4 = [DF,DF2]
# df3 = pd.concat(df4)

# print(df3)

# dataframe = pd.DataFrame({"Actuals":DF, "Predicted":DF2})
# print(dataframe)
# confusion_matrix=confusion_matrix(dataframe["Actuals"], dataframe["Predicted"])
# print(confusion_matrix)
# FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
# FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
# TP = np.diag(confusion_matrix)
# TN = confusion_matrix.sum() - (FP + FN + TP)
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP.sum()/(TP+FN).sum()
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP.sum()/(FP+TN).sum()
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
# # Overall accuracy
# ACC = (TP+TN).sum()/(TP+FP+FN+TN).sum()
# # Recall
# Recall = (TP.sum()/(FN+TP).sum())
# # Specificity
# Specificity = 1-FPR
# print(ACC)
# print(Recall)
# print(Specificity)