# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:01:08 2019

@author: jizh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:33:46 2019

@author: jizh
"""
import cv2
import h5py
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.color import colorconv
import colour
from spectral import *
from PIL import Image
from matplotlib import cm
from matplotlib import colors
from simple_imageWhiteBalanceCorrection import simplest_cb


path = "input_path"
train_ids = next(os.walk(path))[2]
train_ids

##  try to segment image based on color
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    print(n)
    nemo = cv2.imread(os.path.join(path, id_))
    nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
    nemo_x = nemo[:int(nemo.shape[0]*0.85),:,:]
    hsv_nemo = cv2.cvtColor(nemo_x, cv2.COLOR_RGB2HSV)

    ## bundaries determination
    # first blue
    nemo_rz = resize(nemo_x, (800, 700), mode='constant', preserve_range=True).astype(np.uint8)
    hsv_nemo_rz = cv2.cvtColor(nemo_rz, cv2.COLOR_RGB2HSV)
    
    plt.figure(figsize = (7,7))
    plt.imshow(hsv_nemo_rz)
    plt.show()


    for x in range(4):
        b_blue_i = cv2.selectROI(id_, hsv_nemo_rz)
        b_blue_i_p = np.array(hsv_nemo_rz[int(b_blue_i[1]), int(b_blue_i[0])]).astype(np.uint8)
        b_blue_i_p= b_blue_i_p.reshape((1,3))
        print(b_blue_i_p)
        if x ==0:
            print(x)
            b_blue_i_x = b_blue_i_p.copy()
            print(b_blue_i_x)
        else:
            b_blue_i_x = np.concatenate((b_blue_i_x, b_blue_i_p), 0)
        
        b_blue_i_p_low = np.array([np.min(b_blue_i_x[:,0])-10,np.min(b_blue_i_x[:,1])-10,np.min(b_blue_i_x[:,2])-10]).astype(np.uint8)
        b_blue_i_p_low[b_blue_i_p_low<0]=0
        b_blue_i_p_high = np.array([np.max(b_blue_i_x[:,0])+10,np.max(b_blue_i_x[:,1])+10,np.max(b_blue_i_x[:,2])+10])
        b_blue_i_p_high[b_blue_i_p_high>255]=255
        b_blue_i_p_high = b_blue_i_p_high.astype(np.uint8)
        ## for leave selection
    for x in range(10):
        print("warning, time to select leaf pixels")
        l_green_i = cv2.selectROI(id_, hsv_nemo_rz)
        l_green_i_p = np.array(hsv_nemo_rz[int(l_green_i[1]), int(l_green_i[0])]).astype(np.uint8)
        l_green_i_p= l_green_i_p.reshape((1,3))
        print(l_green_i_p)
        if x ==0:
            print(x)
            l_green_i_x = l_green_i_p.copy()
            print(l_green_i_x)
        else:
            l_green_i_x = np.concatenate((l_green_i_x, l_green_i_p), 0)
        
        l_green_i_p_low = np.array([np.min(l_green_i_x[:,0])-10,np.min(l_green_i_x[:,1])-10,np.min(l_green_i_x[:,2])-10]).astype(np.uint8)
        l_green_i_p_low[l_green_i_p_low<0]=0
        l_green_i_p_high = np.array([np.max(l_green_i_x[:,0])+10,np.max(l_green_i_x[:,1])+10,np.max(l_green_i_x[:,2])+10])
        l_green_i_p_high[l_green_i_p_high>255]=255
        l_green_i_p_high = l_green_i_p_high.astype(np.uint8)

    light_blue, dark_blue = b_blue_i_p_low, b_blue_i_p_high

    light_blue = np.array(light_blue, dtype = "uint8")
    dark_blue = np.array(dark_blue, dtype = "uint8")

    mask_ref = cv2.inRange(hsv_nemo, light_blue, dark_blue)
    result_ref = cv2.bitwise_and(nemo_x, nemo_x, mask=mask_ref)

    plt.figure(figsize = (9,12))
    plt.imshow(mask_ref)
    plt.show()

    #############
    # fill the holes of reference
    kernel = np.ones((5,5),np.uint8)
    mask_ref = cv2.dilate(mask_ref,kernel,iterations=5)
    plt.figure(figsize = (6,6))
    plt.imshow(mask_ref,cmap='nipy_spectral')

    contours_ref,hier = cv2.findContours(mask_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#    create hull array for convex hull points
    hull_ref = []
    # calculate points for each contour
    for i in range(len(contours_ref)):
        # creating convex hull object for each contour
        if cv2.contourArea(contours_ref[i])>1000: # neglect noises
            hull_ref.append(cv2.convexHull(contours_ref[i], False))
    co_ref_area = []
    for contour in contours_ref:
        co_ref_area.append(cv2.contourArea(contour))
    co_ref_area_a= np.array(co_ref_area)
    np.where(co_ref_area_a==max(co_ref_area_a))
    
    cnt_ref = contours_ref[np.where(co_ref_area_a==max(co_ref_area_a))[0][0]] ## choose the 
    hull_ref_x = cv2.convexHull(cnt_ref)
    x_ref,y_ref,w_ref,h_ref = cv2.boundingRect(cnt_ref)
    p_lth = 5/w_ref # (cm/pixel)

    rect_ref = cv2.minAreaRect(cnt_ref)
    box_ref = cv2.boxPoints(rect_ref)
    box_ref = np.int0(box_ref)
    cv2.drawContours(nemo_x,[box_ref],0,(0,255,0),4)
#    plt.figure(figsize = (9,9))
#    plt.imshow(nemo_x)
#    plt.show()
    ###               the leaves
    light_green, dark_green = l_green_i_p_low, l_green_i_p_high
    light_green = np.array(light_green, dtype = "uint8")
    dark_green = np.array(dark_green, dtype = "uint8")

    mask_green = cv2.inRange(hsv_nemo, light_green, dark_green)
    #mask = cv2.inRange(hsv_nemo, dark_orange, light_orange)
    result_green = cv2.bitwise_and(nemo_x, nemo_x, mask=mask_green)
    
    #(mask == 1).sum()
    #plt.subplot(1, 2, 1)
    #plt.imshow(mask, cmap="gray")
    #plt.subplot(1, 2, 2)
    plt.figure(figsize = (9,12))
    plt.imshow(mask_green)
    plt.show()

    ## closing the holes of leaf mask
    #############
    contour_leaf,hier = cv2.findContours(mask_green, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contour_leaf:
        cv2.drawContours(mask_green,[cnt],0,255,-1)
    
    plt.figure(figsize = (6,6))
    plt.imshow(mask_green,cmap='nipy_spectral')

    contours_green,hier = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    co_area = []
    for contour in contours_green:
        co_area.append(cv2.contourArea(contour))
    co_area_a= np.array(co_area)
    np.where(co_area_a==max(co_area_a))[0][0]
    # fine the biggest convex
    cnt_leaf = contours_green[np.where(co_area_a==max(co_area_a))[0][0]]
#    x_leaf,y_leaf,w_leaf,h_leaf = cv2.boundingRect(cnt_leaf)
    ellipse_leaf = cv2.fitEllipse(cnt_leaf)
    ## find the values of short and long axises
    w_leaf = ellipse_leaf[1][0]
    h_leaf = ellipse_leaf[1][1]
    ## the assumption, the intersection of short axis is a circle
    if w_leaf<h_leaf:
        w_leaf_r = w_leaf*p_lth
        h_leaf_r = h_leaf*p_lth
    else:
        w_leaf_r = h_leaf*p_lth
        h_leaf_r = w_leaf*p_lth
    
    # calculate the volume of ellipsoid and save
    lettuce_v_i = (np.pi*w_leaf_r*w_leaf_r*h_leaf_r)/6
    lettuce_v_i_x = np.array([id_.split(".")[0], lettuce_v_i]).reshape((1,2))
    if n ==0:
        lettuce_v = lettuce_v_i_x
    else:
        lettuce_v= np.concatenate((lettuce_v,lettuce_v_i_x),0)
    
    # save the final image
#    ellipse_leaf = cv2.fitEllipse(cnt_leaf)
    nemo_final =cv2.ellipse(nemo_x, ellipse_leaf,(0,0,255),2)
    out_path = path+"_out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    id_x = id_.split(".")[0] +".png"
    nemo_final = cv2.cvtColor(nemo_final, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_path, id_x), nemo_final)

    plt.close("all")
























































