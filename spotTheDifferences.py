import cv2
import imutils
import numpy as np

def spotTheDifferences(origin, level):
    #Đọc ảnh đầu vào
    img = cv2.imread(origin)
    level_img = cv2.imread(level)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    level_img_gray = cv2.cvtColor(level_img, cv2.COLOR_BGR2GRAY)

    #Lấy 2 ảnh trừ đi cho nhau để tìm sự khác biệt
    diff = cv2.absdiff(img_gray, level_img_gray)
    cv2.imshow("Differences", diff)

    #Chuyển ảnh về dạng nhị phân
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Binary Image", thresh)

    #Sử dụng dilation lên ảnh
    kernel = np.ones((9,9), np.uint8)
    dilated_img = cv2.dilate(thresh, kernel, iterations = 2)
    cv2.imshow("Dilated Image", dilated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #find the contours
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw bounders around all differences
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(level_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return level_img

