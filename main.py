import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = FaceDetector(minDetectionCon=0.75)

while True:
    success, img = cap.read()
    img, b_boxes = detector.findFaces(img)

    if b_boxes:
        for i, b_box in enumerate(b_boxes):
            x, y, w, h = b_box['bbox']
            if x < 0: x = 0
            if y < 0: y = 0

            img_crop = img[y:y + h, x:x + w]
            img_blur = cv2.blur(img_crop,(35,35))
            img[y:y + h, x:x + w] = img_blur
            # cv2.imshow(f'image_cropped {i}',img_crop)


    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
