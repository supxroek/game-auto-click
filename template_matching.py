import cv2
import numpy as np

def display_image(title, image, size=(300, 300)):
    cv2.imshow(title, image)
    cv2.resizeWindow(title, *size)
    cv2.waitKey()
    cv2.destroyAllWindows()

def match_template(org_image, template, threshold=0.9):
    w, h = template.shape[1], template.shape[0]
    result = cv2.matchTemplate(org_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    yloc, xloc = np.where(result >= threshold)
    return result, max_loc, max_val, xloc, yloc, w, h

def draw_rectangles(image, locations, w, h, color=(0, 255, 255)):
    for (x, y) in locations:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

org_test = cv2.imread("image/imgtest.png", cv2.IMREAD_UNCHANGED)
wood_left = cv2.imread("image/left.png", cv2.IMREAD_UNCHANGED)
wood_right = cv2.imread("image/right.png", cv2.IMREAD_UNCHANGED)

display_image("Org", org_test)
display_image("Wood Left", wood_left)
display_image("Wood Right", wood_right)

# Wood Right
result, max_loc, max_val, xloc, yloc, w, h = match_template(org_test.copy(), wood_right)
print(max_loc, max_val)
print(len(xloc))
org2 = draw_rectangles(org_test.copy(), zip(xloc, yloc), w, h)
cv2.rectangle(org2, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)
display_image("Result Right", org2)

# Wood Left
result, max_loc, max_val, xloc, yloc, w, h = match_template(org_test.copy(), wood_left)
print(max_loc, max_val)
print(len(xloc))
recs = [[int(x), int(y), int(w), int(h)] for (x, y) in zip(xloc, yloc)]
recs, weights = cv2.groupRectangles(recs * 2, 1, 0.2)
org1 = draw_rectangles(org_test.copy(), recs, w, h)
cv2.rectangle(org1, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)
display_image("Result Left", org1)

