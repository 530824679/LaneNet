import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import glob
import numpy as np

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
obj = np.zeros((5 * 8, 3), np.float32)

# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
obj[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("./image/*.jpg")
i=0;
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
    #print(corners)

    if ret:

        obj_points.append(obj)
        # 在原角点的基础上寻找亚像素角点
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (8, 5), corners, ret)

        i += 1;
        cv2.imwrite('./draw/'+str(i)+'.jpg', img)
        cv2.waitKey(1000)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)        # 内参数矩阵
print("dist:\n", dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)    # 旋转向量
print("tvecs:\n", tvecs )   # 平移向量

print("-----------------------------------------------------")

img = cv2.imread(images[2])
h, w = img.shape[:2]

# 显示更大范围的图片（正常重映射之后会删掉一部分图像）
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print (newcameramtx)
print("------------------使用undistort函数-------------------")

dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x, y, w, h = roi
dst1 = dst[y:y+h,x:x+w]

cv2.imwrite('calibresult3.jpg', dst1)
print ("方法一:dst的大小为:", dst1.shape)