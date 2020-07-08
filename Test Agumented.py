import cv2
import numpy as np

cap = cv2.VideoCapture(1)
imgTarget = cv2.imread('Resources/Modi.JPG')                       #Modi.JPG
myVid = cv2.VideoCapture('Resources/Modi_Speech.mp4 ')                     #Modi_Speech.mp4

detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape #Height, Width, Channel
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=1000) #Oriented FAST and Rotated BRIEF Detector
kp1, des1 = orb.detectAndCompute(imgTarget, None)


# imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)

while True:

    sucess, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    if detection == False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    if len(good) > 20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)  #use a function from calib3d module, ie cv2.findHomography()
                                                                            # RANSAC Separates Inliers and Outliers
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8) # create blank image of Augmentation size
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255)) # fill the detected area with white pixels to get mask
        maskInv = cv2.bitwise_not(maskNew) # get inverse mask
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv) # make augmentation area black in final image
        imgAug = cv2.bitwise_or(imgWarp, imgAug) # add final image with warped image
        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(imgAug, 'FPS: {} '.format(int(fps)), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 20, 20),3);
        cv2.putText(imgAug, 'Target Found: {} '.format(detection), (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(230, 20, 20), 3);
        #imgStacked = stackImages(([imgWebcam, imgVideo, imgTarget], [imgFeatures, imgWarp, imgAug]), 0.5)

    cv2.imshow('maskNew', imgAug)
    cv2.imshow('imgWarp', imgWarp)
    cv2.imshow('img2', img2)
    cv2.imshow('imgFeatures', imgFeatures)
    cv2.imshow('ImgTarget',imgTarget)
    cv2.imshow('myVid',imgVideo)
    cv2.imshow('Webcam', imgWebcam)
    #cv2.imshow('MaskNew', maskNew)
    #cv2.imshow('maskInv', maskInv)
    #cv2.imshow('imgStacked', imgStacked)
    cv2.waitKey(1)
    frameCounter += 1
