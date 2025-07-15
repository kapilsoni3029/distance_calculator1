
import cv2
import numpy as np
import math as m

distance_threshold=0.06912


frame=cv2.VideoCapture(0)

while True:
    ok,img=frame.read()
    
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_yellow=np.array([20,100,100])
    upper_yellow=np.array([30,255,255])

    mask=cv2.inRange(src=hsv,lowerb=lower_yellow,upperb=upper_yellow)

    result=cv2.bitwise_and(src1=img,src2=img,mask=mask)

    contors,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    le=65
    XY=[]

    for i in contors:
        if cv2.contourArea(i)>500:
            text=chr(le)
            le=le+1
            x,y,w,h=(cv2.boundingRect(i))
    
            cx=x+(w)//2
            cy=y+(h)//2
        
            XY.append([cx,cy])
        
            cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),3)
            cv2.circle(img,(cx,cy),(5),(0,0,255),-1)
        
    

            cv2.putText(img,f'{text}',((cx-50),(cy+10)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
    

    
    for i in range(len(XY)-1):
        x1,y1=XY[i]
        x2,y2=XY[i+1]
        distance=m.sqrt((x2-x1)**2+(y2-y1)**2)
        
        tx=(x1+x2)//2
        ty=(y1+y2)//2
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(img,f'{distance*(distance_threshold):.2f} cm',(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)

        

    cv2.imshow("hsv",hsv)

    cv2.imshow("mask",mask)


    cv2.imshow("result",result)
    cv2.imshow("final output",img)



    if cv2.waitKey(1)==ord("q"):
        break
    

cv2.destroyAllWindows()
