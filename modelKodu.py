import tensorflow as tf
import keras
import cv2
from keras.models import Sequential
import keras
from keras.layers.convolutional import Convolution2D,MaxPooling2D
import numpy as np
model = tf.keras.models.load_model(
    'tenRengi.h5',compile=True
)







def tahminEt(fotograf):
    boyutlandilirmisGoruntu = cv2.resize(fotograf,(300,300),interpolation=cv2.INTER_AREA)
    image = np.expand_dims(boyutlandilirmisGoruntu, axis=0)
    tahmin=model.predict(image)
    yazi=''
    #tahmin[0][0]+=0.2
    if tahmin[0][0]<tahmin[0][1]:
        return 'Esmer',tahmin
    else:
        return 'Sarisin',tahmin
while True:
    kamera=cv2.VideoCapture(0) # 0 numaralı kayıtlı kamerayı alma
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT,480) #­ 
    

    ret,goruntu=kamera.read() # kamera okumayı başlatma
    
    yazi,tahmin=tahminEt(goruntu)    
    
    cv2.putText(goruntu,yazi,(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255,255),2,cv2.LINE_4);
    print(tahmin)
    cv2.imshow('Normal Goruntu',goruntu)
    cv2.waitKey(40)

cv2.destroyAllWindows()