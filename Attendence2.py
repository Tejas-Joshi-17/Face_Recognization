from email import message
import cv2
import numpy as np
from datetime import date
today = date.today()
from datetime import datetime
now = datetime.now()

d1 = today.strftime("%d/%m/%y")

curent_time = now.strftime("%H:%M:%S")
import pandas as pd
import face_recognition
import os
import smtplib as s
from datetime import datetime

ob=s.SMTP("smtp.gmail.com",587)
ob.starttls()

ob.login("tejas.22010508@viit.ac.in","28-@-2002")
subject = "Email-Message of Attendence system useing face-recognition"
body = ("Your Attendence is Marked Today :- ",d1, curent_time)
message = "Subject:{}\n\n{}".format(subject,body)
listofAddress =["joshitejas188@gmail.com","harshal.22010399@viit.ac.in","shubham.22010275@viit.ac.in"]
# print(message)


path = 'images'
images = []
personNames = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis =face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()

            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            if(name == 'TEJAS'):
                ob.sendmail("tejas.22010508@viit.ac.in",listofAddress[0],message)
                print("Send Mail to Tejas Succesfully")
            if(name == 'HARSHAL'):
                ob.sendmail("tejas.22010508@viit.ac.in",listofAddress[1],message)
                print("Send Mail to Harshal Succesfully")
            if(name == 'SHUBHAM'):
                ob.sendmail("tejas.22010508@viit.ac.in",listofAddress[2],message)
                print("Send Mail to Shubham Succesfully")
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

readfile = pd.read_csv('attendance.csv')
excelwriter = pd.ExcelWriter('attendance.xlsx')
readfile.to_excel(excelwriter)
excelwriter.save()
 
ob.quit()


cap.release()
cv2.destroyAllWindows()
