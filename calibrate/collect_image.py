import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

cap = cv2.VideoCapture(0)

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024, 512))
    cv2.imshow("capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite('./image/'+str(count) + '.jpg', frame)
        count = count + 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()