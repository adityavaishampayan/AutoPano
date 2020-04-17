#!/usr/bin/env python
#reference github.com: Pratiquea
# coding: utf-8

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano

Author(s):
Aditya Vaishampayan (adityav@terpmail.umd.edu)
Hetansh Patel (hpatel57@umd.edu)
"""

import numpy as np
import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
from matplotlib import pyplot as plt
import math
import numpy.linalg
import glob
import copy
import skimage.feature
import argparse


# In[3]:

def ransac(pairs, N,t,thresh):
    M = pairs
    H_new = np.zeros((3,3))
    max_inliers = 0

    for j in range(N):

        index = []

        pts = [np.random.randint(0,len(M)) for i in range(4)]

        a = [M[pts[0]][0][1:3]]

        b = [M[pts[0]][1][1:3]]

        c = [M[pts[1]][0][1:3]]

        d = [M[pts[1]][1][1:3]]

        e = [M[pts[2]][0][1:3]]

        f = [M[pts[2]][1][1:3]]

        g = [M[pts[3]][0][1:3]]

        h = [M[pts[3]][1][1:3]]


        p1 = np.array([a,c,e,g],np.float32)
        p2 = np.array([b,d,f,h],np.float32)

        Homography = cv2.getPerspectiveTransform( p1, p2 )

        inLiers = 0

        for index in range(len(M)):

            dst = np.array(M[index][1][1:3])

            src = np.array(M[index][0][1:3])

            prediction = np.matmul(Homography, np.array([src[0],src[1],1]))

            if prediction[2] == 0:

                prediction[2] = 0.000001

            pred_x = prediction[0]/prediction[2]

            pred_y = prediction[1]/prediction[2]

            prediction = np.array([pred_x,pred_y])

            prediction = np.float32([point for point in prediction])

            if thresh > (np.linalg.norm(dst-prediction)) :

                inLiers += 1

                index.append(ind)

        points1 = []

        points2 = []

        if inLiers > max_inliers:

            max_inliers = inLiers

            [pts1.append([M[i][0][1:3]]) for i in index]

            [pts2.append([M[i][1][1:3]]) for i in index]

            pts1 = np.float32(points1)

            pts2 = np.float32(points2)

            new_homography,status = cv2.findHomography(pts1, pts2)

            if inLiers > t*len(M):
                print('success')

                break

    pairs = [M[i] for i in index]

    if len(pairs)<=4:
        print('Number of pairs after RANSAC is low')

    return new_homography,pairs


# Create Feature Descriptor
def feature_desciptor(img, anms_out):

    pad_width = patch_size = 40
    feats = []

    if (patch_size%2) != 0:
        print('Patch Size should be even')
        return -1

    length, width = np.shape(anms_out)

    img_pad = np.pad(img,(patch_size),'constant',constant_values=0)

    rows = (patch_size/5)**2

    descriptor = np.array(np.zeros((int(rows),1)))

    for i in range(length):

        y1 = anms_out[i][2]+(patch_size/2)
        y2 = anms_out[i][2]+(3*patch_size/2)
        x1 = anms_out[i][1]+(patch_size/2)
        x2 = anms_out[i][1]+(3*patch_size/2)

        y1 = int(y1)
        y2 = int(y2)
        x1 = int(x1)
        x2 = int(x2)

        patch = img_pad[y1:y2,x1:x2]

        blur_patch = cv2.GaussianBlur(patch,(5,5),0)

        sub_sample = blur_patch[0::5,0::5]

        cv2.imwrite('./patches/patch'+str(i)+'.png',sub_sample)

        feats = sub_sample.reshape(int((patch_size/5)**2),1)

        #make the mean 0
        feats = feats - np.mean(feats)

        #make the variance 1
        feats = feats/np.std(feats)
        cv2.imwrite('./features/feature_vector'+str(i)+'.png',feats)
        descriptor = np.dstack((descriptor,feats))

    return descriptor[:,:,1:]


# In[5]:


def feature_matching(descriptor1,descriptor2,kp1,kp2):

    kp1 = best_corners1

    kp2 = best_corners2

    features1 = descriptor1

    features2 = desriptor2

    m,y,n = np.shape(features2)

    p,x,q = np.shape(features1)

    minimum = min(q,n)

    q = int(minimum)

    maximum = max(q,n)

    n = int(maximum)

    matchPairs = []

    q = int(min(q,n))

    n = int(max(q,n))

    for i in range(q):
        #initilaising an empty dictionary

        match_pairs = {}
        for j in range(n):

            value = (features1[:,:,i]-features2[:,:,j])

            ssd = np.linalg.norm(value)**2
            match_pairs[ssd] = [best_corners1[i,:],best_corners2[j,:]]

        S = sorted(match)

        s0 = S[0]

        s1 = S[1]

        if s0/s1 < 0.7:

            pairs = match[first]
            matchPairs.append(pairs)

    return matchPairs


# In[6]:

def stitching_pano(image, homography,image2_shape):
    '''
    image is the input image to be warped
    homography estimated using Ransac
    '''
    h, w, z = np.shape(image)

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    primep = np.dot(homography, p)

    a = primep[0]
    b = primep[1]
    c = primep[2]


    row_y = b / c
    row_x = a / c

    max_x = max(row_x)

    min_y = min(row_y)

    max_y = max(row_y)

    min_x = min(row_x)

    new_mat = np.array([[1, 0, -1 * min_x], [0, 1, -1 * min_y], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    alpha = round(max_y - min_y)

    beta = round(max_x - min_x)

    h = int(alpha)+image2_shape[0]
    w = int(beta)+ image2_shape[1]

    size = (h,w)

    warp = cv2.warpPerspective(src=image, M=homography, dsize=size)

    return warp, int(xmin), int(ymin)



def AdapNonMaxSupression(gray_image, numcorners):

    num_corners = 1000
    quality_level = 0.01
    minimum_eucledian_dist = 10

    p,_,q = features.shape
    r = 1e+8 *np.ones([p,3])
    ed = 0

    corners = cv2.goodFeaturesToTrack(gray_image, num_corners, quality_level,minimum_eucledian_dist)

    for i in range(p):
        for j in range(p):

            y_j = int(corners[j,:,1])
            x_j = int(corners[j,:,0])
            y_i = int(corners[i,:,1])
            x_i = int(corners[i,:,0])


            if gray[y_j,x_j] < gray[y_i,x_i]  :

                ed = (y_j-y_i)**2 + (x_j-x_i)**2

            if r[i,0] > ed:

                r[i,0] = ed
                r[i,1] = x_i
                r[i,2] = y_i

    features = r[np.argsort(-r[:, 0])]

    optimal_corners = features[:numcorners,:]

    return optimal_corners


# In[4]:

def h_estimate(img1,img2):

    flag = True

    image1 = img1
    image2 = img2

    gray_im_1, gray_im_2 = gray_images(image1,image2)

    gray1 = gray_im_1
    gray2 = gray_im_2

    """
    Corner Detection
    """
    corner_detection(gray1,image1,1)
    corner_detection(gray2,image2,2)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    """
    green = (0,255,0)

    number_of_corners = 700

    best_corners1 = AdapNonMaxSupression(gray_im_1, number_of_corners)
    anms1 = copy.deepcopy(img1)
    for corner1 in best_corners1:
        unwanted,x_1,y_1 = corner1.ravel()
        cv2.circle(anms1,(int(x_1),int(y_1)),3,green,-1)

    cv2.imwrite('anms1.png',anms1)

    best_corners2 = AdapNonMaxSupression(gray_im_2, number_of_corners)
    anms2 = copy.deepcopy(img2)
    for corner2 in best_corners2:
        _,x_2,y_2 = corner2.ravel()
        cv2.circle(anms2,(int(x_2),int(y_2)),3,green,-1)

    cv2.imwrite('anms2.png',anms2)


    """
    Feature Descriptors
    """
    feat1 = feature_desciptor(img=gray_im_1, anms_out=best_corners1)
    feat2 = feature_desciptor(img=gray_im_2, anms_out=best_corners2)


    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    matchPairs = feature_matching(feat1,feat2,best_corners1,best_corners2)

    #print("Number of matches",len(matchPairs))

    if len(matchPairs)<45:
        print('Error')
        flag = False

    displayFeatures(img1,img2,matchPairs,new_img_name = 'matching.png')


    """
    Refine: RANSAC, Estimate Homography
    """
    Hmg,pairs = ransac(matchPairs,N=3000,t=0.9 ,thresh=30.0)

    displayFeatures(img1,img2,pairs,new_img_name = 'ransac.png')

    # Hmg = cv2.findHomography(matchPairs[],)
    return Hmg,flag


# In[7]:




# In[8]:



# In[ ]:


def gray_images(img1,img2):

    img1 = img1.astype('np.uint8')
    img2 = img2.astype('np.uint8')


    gray_img_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img_1 = np.float32(gray_img_1)

    gray_img_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray_img_2 = np.float32(gray_img_2)


    return gray_img_1,gray_img_2


# In[ ]:


def corner_detection(gray_img,image,val):

    num_corners = 100000
    quality_level = 0.001

    minimum_eucledian_dist = 10

    corners = cv2.goodFeaturesToTrack(gray_img,num_corners , quality_level,minimum_eucledian_dist)
    corners = np.int0(corners)

    i1 = copy.deepcopy(image)
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(i1,(x,y),3,(0,255,0),-1)

    cv2.imwrite('corners_' + str(val) + '.png',i1)



# In[9]:





# In[10]:

def displayFeatures(img1,img2,matchPairs,new_img_name):


    image1 = img1
    image2 = img2

    a = image1.shape[0]

    b = image2.shape[0]

    c = image1.shape[1]

    d = image2.shape[1]

    e = image1.shape[2]

    rows = max(a,b)

    if len(image1.shape) == 3:
        new_shape = (rows, c+d, e)

    elif len(image1.shape) == 2:
        new_shape = (rows, c+d)

    new_img = np.zeros(new_shape, type(img1.flat[0]))

    # Place images onto the new image.
    new_img[0:a,0:c] = img1
    new_img[0:b,c:c+d] = img2

    r = 15
    thickness = 2
    c = None

    for i in range(len(matchPairs)):
        x_1 = int(matchPairs[i][0][1])
        x_2 = int(matchPairs[i][1][1])+int(c)

        y_1 = int(matchPairs[i][0][2])
        y_2 = int(matchPairs[i][1][2])

        cv2.line(new_img,(x_1,y_1),(x_2,y_2),(0,0,255),2)
        cv2.circle(new_img,(x_1,y_1),3,(0,255,0),-1)
        cv2.circle(new_img,(x_2,y_2),3,(0,255,0),-1)

    cv2.imwrite(new_img_name,new_img)




def main():


    """
    Argument parsing
    """
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', help='Define Path of the Image Set folder')

    Args = Parser.parse_args()
    BasePath = Args.BasePath

    """
    reading all the images from a file
    """
    filenames = [img for img in glob.glob(str(BasePath)+'/*.jpg')]
    filenames.sort() # ADD THIS LINE
    images = []
    for img in filenames:
        n= cv2.imread(img)
        images.append(n)
        print (img)

    images = [cv2.imread(file) for file in sorted(glob.glob(str(BasePath)+'/*.jpg'))]

    """
    starting the image blending process
    """
    image1 = images[0]
    for image in range(1,len(images)-1):
        Homography,flag = h_estimate(image1,image)

        print("Homography: ",Homography)

        if flag == False:
            print('Required number of matches are less hence stopping')
            break

        holderimg, origin_offset_x,origin_offset_y = stitching_pano(image1,Homography,im.shape)

        oY = abs(origin_offset_y)

        oX = abs(origin_offset_x)


        for y in range(oY,im.shape[0]+oY):
            for x in range(oX,im.shape[1]+oX):

                image2_yval = y - oY
                image2_xval = x - oX

                imgholder[y,x,:] = im[image2_yval,image2_xval,:]

        pano_img1 = holderimg

    panaroma = cv2.GaussianBlur(pano_img1,(5,5),1.2)
    cv2.imwrite('pano.png',panaroma)



# In[13]:


if __name__ == '__main__':
    main()


# In[ ]:
