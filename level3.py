import cv2
import numpy as np
import spotTheDifferences as solve

def level3(img_path, texture_path):
    #Đọc ảnh đầu vào
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Áp dụng Laplacian
    texture_img = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=3)
    
    #Chuyển ảnh về dạng nhị phân và tìm vùng texture
    thresh = cv2.threshold(texture_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[4])

    #Đọc ảnh texture và thay đổi kích cỡ cho vừa với vùng texture lựa chọn
    new_texture = cv2.imread(texture_path)
    new_texture = cv2.cvtColor(new_texture, cv2.COLOR_BGR2GRAY)
    new_texture = cv2.resize(new_texture, (w, h))

    #Đổi texture vùng được chọn
    for i in range(0, h):
        for j in range(0, w):
            if (texture_img[y + i, x + j] == 0).all():
                img[y + i, x + j] = new_texture[i, j]
    
    #Ghi ảnh ra file
    cv2.imwrite('code/spotTheDifferences/result/level3.png', img)

    #Ghi ảnh đáp án ra file
    solved_level3 = solve.spotTheDifferences(img_path, "code/spotTheDifferences/result/level3.png")
    cv2.imwrite("code/spotTheDifferences/solved_images/level3_solved.png", solved_level3)

