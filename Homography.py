import cv2
import numpy as np

# 加载图片A和图片B
imgA = cv2.imread('ori.jpg')  # 图片A
imgB = cv2.imread('screen.jpg')  # 图片B，其中包含显示屏幕

# 定义图片A的四个角（源点）
hA, wA = imgA.shape[:2]
src_points = np.array([[0, 0], [wA, 0], [wA, hA], [0, hA]], dtype=np.float32)

# 手动定义图片B中显示屏幕的四个角（目标点），根据实际屏幕角点位置
# 比如屏幕的四个角分别是屏幕左上、右上、右下、左下的坐标
dst_points = np.array([[396, 80], [539, 32], [539, 302], [396, 302]], dtype=np.float32)

# 计算单应性矩阵H
H = cv2.getPerspectiveTransform(src_points, dst_points)

print(H)

# 对图片A进行透视变换，使其符合图片B中屏幕的形状和位置
warped_imgA = cv2.warpPerspective(imgA, H, (imgB.shape[1], imgB.shape[0]))
cv2.imshow('wrapd_imgA', warped_imgA)

# 创建一个掩码，来标识透视变换后的图像区域（即图片A映射到屏幕的区域）
mask = np.zeros_like(imgB, dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(dst_points), (255, 255, 255))  # 填充屏幕区域为白色

# 反转掩码，用于保留图片B中除屏幕区域以外的部分
mask_inv = cv2.bitwise_not(mask)

# 保留图片B中屏幕以外的部分
imgB_background = cv2.bitwise_and(imgB, mask_inv)

# 将透视变换后的图片A放入屏幕区域
final_image = cv2.add(imgB_background, cv2.bitwise_and(warped_imgA, mask))

# 显示结果
cv2.imshow('Result', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
