import cv2
import numpy as np
import spotTheDifferences as solve

def level2(path):
    #Đọc ảnh đầu vào
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Làm mượt ảnh bằng thuật toán gaussian
    img_blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)

    #Áp dụng thuật toán Canny
    img_canny = cv2.Canny(img_blurred, 120, 180)
    
    #Tìm các đường viền
    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    #Đổi màu các vật
    for contour in contours_sorted[6:10]:
        color = np.random.randint(0, 255, size=3)
        cv2.drawContours(img, [contour], 0, color.tolist(), -1)

    #Ghi ảnh ra file
    cv2.imwrite('code/spotTheDifferences/result/level2.png', img)

    #Ghi đáp án ra file
    solved_level2 = solve.spotTheDifferences(path, "code/spotTheDifferences/result/level2.png")
    cv2.imwrite("code/spotTheDifferences/solved_images/level2_solved.png", solved_level2)



