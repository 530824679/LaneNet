import cv2

cap = cv2.VideoCapture(0)

count = 0
while(True):
    ret, frame = cap.read()
    cv2.imshow("capture", frame)

    if cv2.waitKey(100) & 0xFF == ord('s'):
        cv2.imwrite('image/'+str(count) + '.jpg', frame)
        c = c + 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()