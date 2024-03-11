# 


import cv2
import glob
import os

# folder path
dir_path = r'Caltech'
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

path = "Caltech/*.*"
img_number = 1

img_list = glob.glob(path)

for file in img_list[0:count]:
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 1)  #now, we can read each file since we have the full path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w] 
        resized = cv2.resize(roi_color, (125,125))
        cv2.imwrite("extracted/"+str(img_number)+".jpg", resized)
    except:
        print("No faces detected")
    
    
    img_number +=1     




