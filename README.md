# Multi-Object-detection
Multi object detection using mobilenet weights
import cv2 # import opencv
import matplotlib.pyplot as plt
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file) # after dnn press tab this will give you options to comands
classLabels = [] # empty list of python
file_name = 'coco_labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())
print("the number of classes are", len(classLabels))

i = 1
print("class number - class name")

while i <= len(classLabels):
    print(i,"-", classLabels[i-1])
    i = i+1
    
model.setInputSize(320,320) # as model is 320x320 in configuration file
model.setInputScale(1.0/127.5) ## 355/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) ## mobilenet takes input as [-1,1]
model.setInputSwapRB(True) # so automatic conversion from BGR (OpenCV default) to RGB

# read an image
img = cv2.imread('man with car.jpg')
plt.imshow(img) # BGR image
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
ClassIndex,confidence, bbox = model.detect(img,confThreshold = 0.5) # as model is already loaded, and it has 3 outputs
# since confidence level taken by us is 50% so 0.5, this parameter can be changed
print("The classes identified are",ClassIndex , " and the confidence level is", confidence, "respectively")
font_scale = 10
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf,boxes in zip (ClassIndex.flatten(), confidence.flatten(),bbox): # zip is used as 3 diff variables
    cv2.rectangle(img,boxes,(255,0,0),2) # rectangle is box
    text_print = classLabels[ClassInd-1] + " - " + str(round(conf, 2))
    cv2.putText(img,text_print,(boxes[0]-10,boxes[1]-40), font , fontScale = font_scale,color = (0,0,255),thickness=3)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# Video demo
cap = cv2.VideoCapture("London walk.mp4")



#check if video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open Video")
        

font_scale = 1
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret,frame = cap.read()
    
    ClassIndex,confidence,bbox = model.detect(frame,confThreshold = 0.55)
    
    print(ClassIndex)
    if (len(ClassIndex)!=0):
        for ClassInd, conf,boxes in zip (ClassIndex.flatten(), confidence.flatten(),bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                text_print = classLabels[ClassInd-1] + " - " + str(round(conf, 2))
                cv2.putText(frame,text_print,(boxes[0]-10,boxes[1]-40), font , 
                            fontScale = font_scale,color = (0,0,255),thickness=1)
                
                
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(2) & 0xFF ==('q'):
        break
        

cap.release()
cv2.destroyAllWindows
