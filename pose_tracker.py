# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp

def read_posetrack_keypoints(openpose_json_file):

    people = dict()

    json_file = openpose_json_file
    data = json.load(open(json_file))
    for person in data['people']:
        person_id = person['person_id'][0]
        joints2d  = person['pose_keypoints_2d']
        if person_id in people.keys():
            people[person_id]['joints2d'].append(joints2d)
            people[person_id]['frames'].append(idx)
        else:
            people[person_id] = {
                'joints2d': [],
                'frames': [],
            }
            people[person_id]['joints2d'].append(joints2d)
            people[person_id]['frames'].append(0)

    for k in people.keys():
        people[k]['joints2d'] = np.array(people[k]['joints2d']).reshape((len(people[k]['joints2d']), -1, 3))
        people[k]['frames'] = np.array(people[k]['frames'])

    return people