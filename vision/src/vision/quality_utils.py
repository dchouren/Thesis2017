import cv2
import numpy as np
import math

def get_laplacian(im):
    return cv2.Laplacian(im, cv2.CV_64F)

''' lapaclian_array should be the result of get_laplacian '''
def get_blurriness(laplacian_array):
    count = len(laplacian_array[np.where(np.absolute(laplacian_array) > 128)])

    return count / (laplacian_array.shape[0] * laplacian_array.shape[1]) * 100

''' lapaclian_array should be the result of get_laplacian '''
def get_cleanness(laplacian_array):
    count = len(laplacian_array[np.where(np.absolute(laplacian_array) < 8)])

    return count / (laplacian_array.shape[0] * laplacian_array.shape[1]) * 100


def has_face_from_path(im_path, face_cascade):
    try:
        image = np.load(im_path)
    except:
        image = cv2.imread(im_path)
    return has_face(image, face_cascade)


def has_face(cv2_im, face_cascade):
    # gray = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        cv2_im,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
        flags = 0
        # flags = cv2.CASCADE_SCALE_IMAGE
    )

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(cv2_im, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if len(faces) > 0:

        # cv2.imshow("Faces found", cv2_im)
        # cv2.waitKey(0)
        return True
    return False


def get_entropy(cv2_im):
    hist,bins = np.histogram(cv2_im.ravel(),256,[0,256])
    hist_prob = hist / sum(hist)
    entropy = -sum([p * math.log(p, 2) for p in hist_prob if p != 0])
    return entropy