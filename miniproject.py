import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

m_new = tf.keras.models.load_model('model_cloth (1).h5')


img = np.ones([400,400],dtype ='uint8')*255

img[50:350,50:350]=0
wname = 'Canvas'
cv2.namedWindow(wname)
def shape(event,x,y,flags,param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state=True
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        print(x,y)
    if event == cv2.EVENT_MOUSEMOVE:
        if state==True:
            cv2.circle(img,(x,y),10,(255,0,0),-1)
            print(x,y)
        else: state=False

cv2.setMouseCallback(wname,shape)  # Shape is a sub method called in the setMouseCallback method

while True:
    cv2.imshow(wname,img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        img[50:350,50:350]=0
    elif key == ord('w'):
        out = img[50:350,50:350]
        cv2.imwrite('Output2.jpg',out)
        break
img=cv2.imread('Output2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)
img_resize = cv2.resize(binary,(28,28)).reshape(1,28,28)
y=m_new.predict_classes(img_resize)
print(y)

cv2.destroyAllWindows()


