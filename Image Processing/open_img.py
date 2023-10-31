import cv2

img=cv2.imread('puppy_image.jpg')

#img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

while True:
    cv2.imshow('puppy', img)
    # waited 1msec AND we've pressed 'q' key 
    if cv2.waitKey(1) & (0xFF ==ord('q')):     
        break;
        
cv2.destroyAllWindows() 



