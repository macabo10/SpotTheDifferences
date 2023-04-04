import cv2
import numpy as np
import spotTheDifferences as solve

def level1(path):
    #Đọc ảnh đầu vào
    img = cv2.imread(path)

    #Chọn 2 vùng ngẫu nhiên và thay đổi màu
    h, w, _ = img.shape
    for i in range(2):
        #Chọn điểm ngẫu nhiên
        x = np.random.randint(0, h - 30)
        #Chọn màu ngẫu nhiên
        color = np.random.randint(0, 255, size=3)
        #Đổi màu vùng được chọn kích cỡ 30*30
        img[x:x+30, x:x+30] = color

    #Ghi ảnh ra file
    cv2.imwrite("code/spotTheDifferences/result/level1.png", img)

    #Ghi ảnh đáp án ra file
    solved_level1 = solve.spotTheDifferences(path, "code/spotTheDifferences/result/level1.png")
    cv2.imwrite("code/spotTheDifferences/solved_images/level1_solved.png", solved_level1)


