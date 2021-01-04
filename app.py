#!/usr/bin/python
# -*- coding: utf-8 -*-

# <img src="{{ user_image }}" alt="User Image">

from imageai.Detection import ObjectDetection
from PIL import Image
import glob
import os
import flask
from flask import Flask, redirect, url_for, request, render_template
from flask import jsonify
from flask_caching import Cache
from flask_uploads import UploadSet, configure_uploads, ALL
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import base64
import random

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

from random import randint

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

# cache.init_app(app)........

files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = './static/uploaded_imgs'
configure_uploads(app, files)

# OUTPUT_PATH = './static/result_imgs/detected'

roots_to_clear = [app.config['UPLOADED_FILES_DEST'],
                  './static/result_imgs']

# PEOPLE_FOLDER = os.path.join('static', 'people_photo')

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

protoFile = 'pose_deploy_linevec.prototxt'
weightsFile = 'pose_iter_440000.caffemodel'
nPoints = 18

# COCO Output Format

keypointsMapping = [
    'Nose',
    'Neck',
    'R-Sho',
    'R-Elb',
    'R-Wr',
    'L-Sho',
    'L-Elb',
    'L-Wr',
    'R-Hip',
    'R-Knee',
    'R-Ank',
    'L-Hip',
    'L-Knee',
    'L-Ank',
    'R-Eye',
    'L-Eye',
    'R-Ear',
    'L-Ear',
    ]

POSE_PAIRS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 17],
    [5, 16],
    ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.

mapIdx = [
    [31, 32],
    [39, 40],
    [33, 34],
    [35, 36],
    [41, 42],
    [43, 44],
    [19, 20],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [47, 48],
    [49, 50],
    [53, 54],
    [51, 52],
    [55, 56],
    [37, 38],
    [45, 46],
    ]

colors = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0],
    ]


# Find the Keypoints using Non Maximum Suppression on the Confidence Map

@app.route('/', methods=['GET', 'POST'])
def index():
    for root_to_clear in roots_to_clear:
        for (root, dirs, files) in os.walk(root_to_clear):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    return render_template('home.html')


@app.route('/preview', methods=['GET', 'POST'])
def upload_and_preview():
    if request.method == 'POST' and 'media' in request.files:
        _ = files.save(request.files['media'])
        list_of_files = \
            glob.glob(os.path.join(app.config['UPLOADED_FILES_DEST'],
                      '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        global image_to_process
        image_to_process = latest_file

        # print image_to_process
        # print latest_file

        img_to_render = os.path.join('..', latest_file)
        img_to_render = img_to_render.replace('\\', '/')
    else:
        img_to_render = '../static/images/no_image.png'
    paths = ['../static/images/no_image.png']

    return render_template('preview.html', img_name=img_to_render,
                           img_paths=paths)


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def obj_detect():
    apple = random.randint(0, 100000000)
    naming = 'static/people_photo/' + str(apple) + '.png'
    naming_indx = str(apple) + '.png'

    # print naming
    # print naming_indx

    image1 = cv2.imread(image_to_process)

    #print ('image..........', image1)
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    # Fix the input Height and get the width according to the Aspect Ratio

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    inHeight = 368
    inWidth = int(inHeight / frameHeight * frameWidth)

    inpBlob = cv2.dnn.blobFromImage(
        image1,
        1.0 / 255,
        (inWidth, inHeight),
        (0, 0, 0),
        swapRB=False,
        crop=False,
        )

    net.setInput(inpBlob)

    output = net.forward()

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1],
                             image1.shape[0]))

    #     plt.figure()
    #     plt.imshow(255*np.uint8(probMap>threshold))

        # keypoints = getKeypoints(probMap, threshold)

        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []

        # find the blobs

        (contours, _) = cv2.findContours(mapMask, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima

        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]], ))

        # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))

        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id, ))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frameClone = image1.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(
                frameClone,
                (detected_keypoints[i][j])[0:2],
                3,
                [0, 0, 255],
                -1,
                cv2.LINE_AA,
                )

    # plt.figure(figsize=[15,15])
    # plt.imshow(frameClone[:,:,[2,1,0]])

    # (valid_pairs, invalid_pairs) = getValidPairs(output)

    # ---------

    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7

    # loop for every POSE_PAIR

    for k in range(len(mapIdx)):

        # A->B constitute a limb

        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if nA != 0 and nB != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):

                    # Find d_ij

                    d_ij = np.subtract((candB[j])[:2], (candA[i])[:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue

                    # Find p(u)

                    interp_coord = list(zip(np.linspace(candA[i][0],
                            candB[j][0], num=n_interp_samples),
                            np.linspace(candA[i][1], candB[j][1],
                            num=n_interp_samples)))

                    # Find L(p(u))

                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])),
                                int(round(interp_coord[k][0]))],
                                pafB[int(round(interp_coord[k][1])),
                                int(round(interp_coord[k][0]))]])

                    # Find E

                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair

                    if len(np.where(paf_scores > paf_score_th)[0]) \
                        / n_interp_samples > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1

                # Append the connection to the list

                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3],
                            candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list

            valid_pairs.append(valid_pair)
        else:

              # If no keypoints are detected

            # print 'No Connection : k = {}'.format(k)

            invalid_pairs.append(k)
            valid_pairs.append([])

    # print valid_pairs
    # return (valid_pairs, invalid_pairs)
    # --------

    # personwiseKeypoints = getPersonwiseKeypoints(valid_pairs,
    #        invalid_pairs)

    # =======

    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            (indexA, indexB) = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += \
                        keypoints_list[partBs[i].astype(int), 2] \
                        + valid_pairs[k][i][2]
                elif not found and k < 17:

                # if find no partA in the subset, create a new subset

                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]

                    # add the keypoint_scores for the two keypoints and the paf_score

                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :
                                  2].astype(int), 2]) \
                        + valid_pairs[k][i][2]
                    personwiseKeypoints = \
                        np.vstack([personwiseKeypoints, row])

    # ==========

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(
                frameClone,
                (B[0], A[0]),
                (B[1], A[1]),
                colors[i],
                1,
                cv2.LINE_AA,
                )

    plt.figure(figsize=[8, 8])
    plt.imshow(frameClone[:, :, [2, 1, 0]])
    plt.savefig(naming)

    # #### end

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],
                                 naming_indx)
    return render_template('index.html', user_image=full_filename)

if __name__ == '__main__':
    app.run(debug=True)
