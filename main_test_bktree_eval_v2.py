#!/data/anaconda3/bin/python
# -*- coding: utf-8 -*-
#
from pip._internal import main
main(["install","shapely"])
main(["install","Polygon3"])
main(["install","opencv-python"])
main(["install","re"])

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import re
import locality_aware_nms as nms_locality

from bktree import BKTree, levenshtein, list_words

#/data/ceph_11015/ssd/anhan/nba/FOTS_TF/
#/data/ceph_11015/ssd/templezhang/scoreboard/EAST/data/nba_test_1023.csv
tf.app.flags.DEFINE_string('test_data_path', '/data/ceph_11015/ssd/templezhang/scoreboard/EAST/data/check_res_15161718_test_null.csv', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_bool("check_teamname",False, '')
tf.app.flags.DEFINE_bool("just_infer",False, '')
tf.app.flags.DEFINE_string('checkpoint_path','/data/ceph_11015/ssd/anhan/nba/FOTS_TF/checkpoints/bs16_540p_v1106_aughsv/', '')
tf.app.flags.DEFINE_string('output_dir','/data/ceph_11015/ssd/anhan/nba/FOTS_TF/outputs/outputs_bs16_540p_v1106_aughsv_eval', '')
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')
tf.app.flags.DEFINE_string('vocab', '/data/ceph_11015/ssd/anhan/nba/FOTS_TF/vocab.txt', 'strong, normal or weak')

from module import Backbone_branch, Recognition_branch, RoI_rotate
from data_provider.data_utils import restore_rectangle, ground_truth_to_word
FLAGS = tf.app.flags.FLAGS
detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = Recognition_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

all_team=["sanantonio","oklahomacity","utah","washington","philadelphia","phoenix","portland","sacramento","toronto","memphis","atlanta","neworleans","boston","cleveland","chicago","denver","dallas","goldenstate","detroit","indiana","houston","losangeles","la","milwaukee","miami","brooklyn","minnesota","orlando","newyork","charlotte","spurs","thunder","jazz","wizards","76ers","suns","trailblazers","kings","raptors","grizzlies","hawks","pelicans","celtics","cavaliers","bulls","nuggets","mavericks","warriors","pistons","pacers","rockets","lakers","clippers","bucks","heat","nets","timberwolves","magic","knicks","hornets","sas","okc","uta","was","phi","phx","por","sac","tor","mem","atl","nop","bos","cle","chi","den","dal","gsw","det","ind","hou","lal","lac","mil","mia","bkn","min","orl","nyk","cha","trail blazers","sa","phoenixsuns","sanantoniospurs","seattlesupersonics","dallasmavericks","sacramentokings","houstonrockets","memphisgrizzlies","lalakers","minnesotatimberwolves","denvernuggets","laclippers","portlandtrailblazers","utahjazz","goldenstatewarriors","neworleanshornets","miamiheat","detroitpistons","bostonceltics","clevelandcavaliers","washingtonwizards","orlandomagic","chicagobulls","philadelphia76ers","indianapacers","newjerseynets","milwaukeebucks","newyorkknicks","torontoraptors","charlottebobcats","atlantahawks","pho","sea","gs","no","nj","ny","wsh"]
quarter=["1st","2nd","3rd","4th","first","second","third","fourth","1stqtr","2ndqtr","3rdqtr","4thqtr","firstqtr","secondqtr","thirdqtr","fourthqtr","1stquarter","2ndquarter","3rdquarter","4thquarter","ot","end","final","overtime","final/ot"]


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def get_image_self(img_base_dir):
    """
    find image in test data file
    :return: list of files found
    """
    files=[]
    corridate_list=[]
    label_list=[]
    lines=open(FLAGS.test_data_path,"r",encoding="utf-8").readlines()
    #print("======",len(lines))
    for line in lines:
        #print(line)
        sp=line.strip().split(",")
        file=os.path.join(img_base_dir,sp[0])
        #print(file)
        # file=sp[0]
        # file=sp[0].strip().replace("_","/").replace("ceph/","ceph_")
        corridate=sp[1]
        label=sp[2]
        if os.path.exists(file):
            files.append(file)
            corridate_list.append(corridate)
            label_list.append(label)
    print("Find {} image".format(len(files)))
    return (files,corridate_list,label_list)


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float32), nms_thres)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def get_project_matrix_and_width(text_polyses, target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width: 
            max_width = width_box 
        """
        mapped_x2, mapped_y2 = (width_box, 0)
        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def bktree_search(bktree, pred_word, dist=5):
    return bktree.query(pred_word, dist)

def contain_eng(str0):
    return bool(re.search('[a-z]', str0))

def left_up(corr_dic):
    corr_dic_key=list(corr_dic.keys())
    corr1=corr_dic_key[0]
    corr2=corr_dic_key[1]
    res={}
    if math.pow(corr_dic[corr1][0]-corr_dic[corr2][0],2) > math.pow(corr_dic[corr1][1]-corr_dic[corr2][1],2):
        if corr_dic[corr1][0]<corr_dic[corr2][0]:
            res["guest"]=corr1
            res["host"]=corr2
            return res
        else:
            res["guest"] = corr2
            res["host"] = corr1
            return res
    else:
        if corr_dic[corr1][1]<corr_dic[corr2][1]:
            res["guest"] = corr1
            res["host"] = corr2
            return res
        else:
            res["guest"] = corr2
            res["host"] = corr1
            return res


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r

def content_sort(team_name,scores,quarter,time1,time2):
    res=["","","","","",""]
    if FLAGS.check_teamname:
        if len(team_name)==2:
            team_location=left_up(team_name)
            res[0]=team_location["guest"]
            res[2]=team_location["host"]
        else:
            print("wrong len(team_name)={}".format(len(team_name)))
    #
    # if len(quarter)==1:
    #     res[4]=list(quarter.keys())[0]
    # else:
    #     print("wrong len(quarter_list)={}".format(len(quarter)))

    if len(scores.keys())==2:
        score_location=left_up(scores)
        res[1]=score_location["guest"].split("_")[0]
        res[3]=score_location["host"].split("_")[0]
    else:
        print("wrong len(scores)={}".format(len(scores)))
    if time1 is not None:
        res[5]=time1.split("_")[0]
    # if time2 is not None:
    #     res.append(time2.split("_")[0])
    # print(res)
    # if res[0]=="":
    #     del res[0]
    # if res[2]=="":
    #     del res[2]
    return ("_").join(res)


def get_content(remainder_attack_time,time_left,team_name, scores, quarter_dict):
    if len(remainder_attack_time.keys()) == 2:
        time_res = left_up(remainder_attack_time)
        return content_sort(team_name, scores, quarter_dict, time_res["guest"], time_res["host"])

    elif len(time_left.keys()) == 2:
        # print(time)
        time_res = left_up(time_left)
        return content_sort(team_name, scores, quarter_dict, time_res["guest"], time_res["host"])

    elif len(remainder_attack_time.keys()) == 1 and len(time_left.keys()) == 1:

        return content_sort(team_name, scores, quarter_dict, list(time_left.keys())[0],
                            list(remainder_attack_time.keys())[0])

    elif (len(scores.keys()) == 3 and len(time_left.keys()) == 1) \
            or (len(scores.keys()) == 3 and len(remainder_attack_time.keys()) == 1):
        big_score = {}
        small_score = ""
        remainder_attack_time_temp = list(time_left.keys())[0] if len(time_left.keys()) == 1 else \
            list(remainder_attack_time.keys())[0]
        for key in scores.keys():
            if int(key.split("_")[0]) > 24:
                big_score[key] = scores[key]
            else:
                small_score = key
        if len(big_score) == 2:
            score_stay = remove_key(scores, small_score)
            return content_sort(team_name, score_stay, quarter_dict, remainder_attack_time_temp,
                                small_score)
        else:
            if len(time_left.keys()) == 1:  # find remain time
                time_key = list(time_left.keys())[0]
                time_stay = time_left
            else:
                time_key = list(remainder_attack_time.keys())[0]
                time_stay = remainder_attack_time
            right_key = []
            for key in scores.keys():
                if scores[key][0] > time_stay[time_key][0]:
                    right_key.append(key)
            if len(right_key) == 1:
                res_score = remove_key(scores, right_key[0])
                return content_sort(team_name, res_score, quarter_dict, remainder_attack_time_temp,
                                    right_key[0])
            elif len(right_key) == 0:
                # print(label)
                print("right key is null")
            elif len(right_key) == 2:
                right1 = right_key[0]
                right2 = right_key[1]
                distance_dic = {}
                distance1 = math.pow(scores[right1][1] - time_stay[time_key][1], 2)
                distance_dic[right1] = distance1
                distance2 = math.pow(scores[right2][1] - time_stay[time_key][1], 2)
                distance_dic[right2] = distance2
                # print(distance_dic)
                distance_sort = sorted(distance_dic.items(), key=lambda item: item[1])
                if distance_sort[-1][-1] < 30:
                    distance_dic2 = {}
                    distance1 = math.pow(scores[right1][0] - time_stay[time_key][0], 2)
                    distance_dic2[right1] = distance1
                    distance2 = math.pow(scores[right2][0] - time_stay[time_key][0], 2)
                    distance_dic2[right2] = distance2
                    min_right = sorted(distance_dic2.items(), key=lambda item: item[1])[0][0]
                    res_score = remove_key(scores, min_right)
                else:
                    min_right = distance_sort[0][0]
                    res_score = remove_key(scores, min_right)

                return content_sort(team_name, res_score, quarter_dict, remainder_attack_time_temp,
                                    min_right)
            else:
                right1 = right_key[0]
                right2 = right_key[1]
                right3 = right_key[2]
                distance_dic = {}
                distance1 = math.pow(scores[right1][1] - time_stay[time_key][1], 2)
                distance_dic[right1] = distance1
                distance2 = math.pow(scores[right2][1] - time_stay[time_key][1], 2)
                distance_dic[right2] = distance2
                distance3 = math.pow(scores[right3][1] - time_stay[time_key][1], 2)
                distance_dic[right3] = distance3
                min_right = sorted(distance_dic.items(), key=lambda item: item[1])[0][0]
                res_score = remove_key(scores, min_right)

                return content_sort(team_name, res_score, quarter_dict, remainder_attack_time_temp,
                                    min_right)

    elif (len(remainder_attack_time.keys()) == 1 and len(time_left.keys()) == 0):
        # return list(remainder_attack_time.keys())[0]
        return content_sort(team_name, scores, quarter_dict, list(remainder_attack_time.keys())[0],
                            None)
    elif len(remainder_attack_time.keys()) == 0 and len(time_left.keys()) == 1:
        return content_sort(team_name, scores, quarter_dict, list(time_left.keys())[0], None)
    else:

        return None

def get_score_info_v2(corridate,label):
    # print(label)
    scores = {}
    score_index=0
    time={}
    time_index=0
    team_name={}
    quarter_dict={}
    remainder_attack_time={}
    remainder_attack_time_index=0

    if int(len(corridate) / 4) == len(label):
        for i in range(len(label)):
            # corridate_label_center[label[i]] = [(int(corridate[4 * int(i) + 0]) + int(corridate[4 * int(i) + 2])) / 2,
            #                                     (int(corridate[4 * int(i) + 1]) + int(corridate[4 * int(i) + 3])) / 2]
            if label[i] in all_team:
                team_name[label[i]]=[(int(corridate[4 * int(i) + 0]) + int(corridate[4 * int(i) + 2])) / 2,
                                              (int(corridate[4 * int(i) + 1]) + int(corridate[4 * int(i) + 3])) / 2]
            if label[i] in quarter:
                quarter_dict[label[i]]=[(int(corridate[4 * int(i) + 0]) + int(corridate[4 * int(i) + 2])) / 2,
                                              (int(corridate[4 * int(i) + 1]) + int(corridate[4 * int(i) + 3])) / 2]
            if label[i].isdigit():
                scores[label[i] + "_" + str(score_index)] = [(int(corridate[4 * int(i) + 0]) + int(corridate[4 * int(i) + 2])) / 2,
                                                  (int(corridate[4 * int(i) + 1]) + int(corridate[4 * int(i) + 3])) / 2]
                score_index += 1
            if ":" in label[i]:
                time[label[i]+"_"+str(time_index)]=  [(int(corridate[4 * int(i) + 0]) + int(corridate[4 * int(i) + 2])) / 2,
                                                  (int(corridate[4 * int(i) + 1]) + int(corridate[4 * int(i) + 3])) / 2]
                time_index+=1
            if "." in label[i] and ":" not in label[i]:
                remainder_attack_time[label[i]+"_"+str(remainder_attack_time_index)]=[(int(corridate[4 * int(i) + 0]) + int(corridate[4 * int(i) + 2])) / 2,
                                                  (int(corridate[4 * int(i) + 1]) + int(corridate[4 * int(i) + 3])) / 2]
                remainder_attack_time_index+=1

    # if len(team_name.keys())!=0 and len(team_name.keys())!=2:
    #     print("wrong len(team_name)={}".format(len(team_name)))
    #     return  None
    # if len(quarter_dict.keys())>1:
    #     print("wrong len(quarter_list)={}".format(len(quarter_dict)))
    #     return None

    if len(remainder_attack_time.keys())==2:
        # print(remainder_attack_time)
        time_res=left_up(remainder_attack_time)
        return content_sort(team_name,scores,quarter_dict,time_res["guest"],time_res["host"])
        # return remainder_attack_time["host"]

    elif len(time.keys())==2:
        # print(time)
        time_res=left_up(time)
        return content_sort(team_name,scores,quarter_dict,time_res["guest"],time_res["host"])
        # time_temp=time["host"]
        # if time_temp.startswith(":") and float(time_temp.split("_")[0].split(":")[1])<=24:
        #     return time_temp
        # return None


    elif len(remainder_attack_time.keys())==1 and len(time.keys())==1:

        return content_sort(team_name,scores,quarter_dict,list(time.keys())[0],list(remainder_attack_time.keys())[0])
        # return  list(remainder_attack_time.keys())[0] if float(list(remainder_attack_time.keys())[0])<=24 else None
         # if len(time.keys())==2:
    #     time_keys=list(time.keys())
    #     time1=time_keys[0]
    #     time2=time_keys[1]
    #     if not time1.startswith(":") and time2.startswith(":"):
    #         remainder_time[time1]=time[time1]
    #     elif time1.startswith(":") and not time2.startswith(":"):
    #         remainder_time[time2]=time[time2]
    #     elif time1.startswith(":") and time2.startswith(":"):
    #         if time1.split("_")[0]==time2.split("_")[0]:
    #             # print(time)
    #             remainder_time[time1]=time[time2]
    # if score_big==2 or len(scores.keys())==2:
    #     return scores
    elif (len(scores.keys())==3 and len(time.keys())==1)\
            or (len(scores.keys()) == 3 and len(remainder_attack_time.keys()) == 1):
        big_score={}
        small_score=""
        remainder_attack_time_temp = list(time.keys())[0] if len(time.keys()) == 1 else list(remainder_attack_time.keys())[0]
        for key in scores.keys():
            if int(key.split("_")[0])>24:
                big_score[key]=scores[key]
            else:
                small_score=key
        if len(big_score)==2:
            # print(scores,big_score)
            score_stay=remove_key(scores,small_score)
            return content_sort(team_name,score_stay,quarter_dict,remainder_attack_time_temp,small_score)
            # return small_score

        else:
            if len(time.keys())==1:               #find remain time
                time_key=list(time.keys())[0]
                time_stay=time
            else:
                time_key = list(remainder_attack_time.keys())[0]
                time_stay=remainder_attack_time
            right_key=[]
            for key in scores.keys():
                if scores[key][0]>time_stay[time_key][0]:
                    right_key.append(key)
            if len(right_key)==1:
                res_score=remove_key(scores,right_key[0])
                return content_sort(team_name,res_score,quarter_dict,remainder_attack_time_temp,right_key[0])
                # return right_key[0] if int(right_key[0].split("_")[0])<=24 else None
            elif len(right_key)==0:
                #print(label)
                print("right key is null")
            elif len(right_key)==2:
                right1=right_key[0]
                right2=right_key[1]
                distance_dic={}
                distance1=math.pow(scores[right1][1]-time_stay[time_key][1],2)
                distance_dic[right1]=distance1
                distance2=math.pow(scores[right2][1]-time_stay[time_key][1],2)
                distance_dic[right2]=distance2
                # print(distance_dic)
                distance_sort= sorted(distance_dic.items(), key=lambda item: item[1])
                if distance_sort[-1][-1]<30:
                    distance_dic2={}
                    distance1 = math.pow(scores[right1][0] - time_stay[time_key][0], 2)
                    distance_dic2[right1] = distance1
                    distance2 = math.pow(scores[right2][0] - time_stay[time_key][0], 2)
                    distance_dic2[right2] = distance2
                    min_right=sorted(distance_dic2.items(),key=lambda  item:item[1])[0][0]
                    res_score=remove_key(scores,min_right)
                else:
                    min_right =distance_sort[0][0]
                    res_score = remove_key(scores, min_right)

                return  content_sort(team_name,res_score,quarter_dict,remainder_attack_time_temp,min_right)
                #                 # if label[-1] != min_right.split("_")[0]:
                #     # print(scores,time_stay)
                #     # print(distance1, distance2)
                #     # print(time_key)
                #     print(("_").join(label), min_right)
                #     pass
            else:
                right1 = right_key[0]
                right2 = right_key[1]
                right3 = right_key[2]
                distance_dic={}
                distance1 = math.pow(scores[right1][1] - time_stay[time_key][1],2)
                distance_dic[right1]=distance1
                distance2 = math.pow(scores[right2][1] - time_stay[time_key][1],2)
                distance_dic[right2] = distance2
                distance3 = math.pow(scores[right3][1] - time_stay[time_key][1],2)
                distance_dic[right3] = distance3
                min_right=sorted(distance_dic.items(),key=lambda item:item[1])[0][0]
                res_score=remove_key(scores,min_right)

                return content_sort(team_name,res_score,quarter_dict,remainder_attack_time_temp,min_right)
                # if label[-1] != min_right.split("_")[0]:
                #     # print(scores,time_stay)
                #     # print(distance1, distance2)
                #     # print(time_key)
                #     print(("_").join(label), min_right)
                #     pass
    elif (len(remainder_attack_time.keys())==1 and len(time.keys())==0 ):
        # return list(remainder_attack_time.keys())[0]
        return  content_sort(team_name,scores,quarter_dict,list(remainder_attack_time.keys())[0],None)
    elif   len(remainder_attack_time.keys()) == 0 and len(time.keys()) == 1:
        return content_sort(team_name,scores,quarter_dict,list(time.keys())[0],None)
        # if list(time.keys())[0].startswith(":"):
        #     return list(time.keys())[0]
        # else:
        #     return None
    elif len(remainder_attack_time.keys())==0 and len(time.keys())==0 and len(scores.keys())==2:
        return content_sort(team_name,scores,quarter_dict,None,None)
    else:
        # print(label)
        return  None

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    bk_tree = BKTree(levenshtein, list_words(FLAGS.vocab))
    # bk_tree = bktree.Tree()

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')

        input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_score, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map, input_transform_matrix, input_box_mask, input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois, input_box_widths)
        _, dense_decode = recognize_part.decode(recognition_logits, input_box_widths)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            # im_fn_list = get_images()
            if FLAGS.just_infer:
                im_fn_list,_,_=get_image_self("/data/ceph_11015/ssd/anhan/nba/video2image")
            else:
                im_fn_list, corridate_list, label_list = get_image_self("/data/ceph_11015/ssd/anhan/nba/video2image")
            wrong=0
            total=0
            for ind,im_fn in enumerate(im_fn_list):
                #print("im_fn:",im_fn)
                im = cv2.imread(im_fn)[:, :, ::-1]
                im = cv2.resize(im, (960, 540))

                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                # im_resized_d, (ratio_h_d, ratio_w_d) = resize_image_detection(im)

                timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
                start = time.time()
                shared_feature_map, score, geometry = sess.run([shared_feature, f_score, f_geometry], feed_dict={input_images: [im_resized]})

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                timer['detect'] = time.time() - start
                start = time.time() # reset for recognition
                res=None
                str_list=[]
                if boxes is not None and boxes.shape[0] != 0:
                    #res_file_path = os.path.join(FLAGS.output_dir,'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                    # res_file_path = os.path.join(FLAGS.output_dir, '{}.txt'.format(os.path.basename(im_fn)))

                    input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                    recog_decode_list = []
                    # Here avoid too many text area leading to OOM
                    for batch_index in range(input_roi_boxes.shape[0] // 32 + 1): # test roi batch size is 32
                        start_slice_index = batch_index * 32
                        end_slice_index = (batch_index + 1) * 32 if input_roi_boxes.shape[0] >= (batch_index + 1) * 32 else input_roi_boxes.shape[0]
                        tmp_roi_boxes = input_roi_boxes[start_slice_index:end_slice_index]

                        boxes_masks = [0] * tmp_roi_boxes.shape[0]
                        transform_matrixes, box_widths = get_project_matrix_and_width(tmp_roi_boxes)
                        #max_box_widths = max_width * np.ones(boxes_masks.shape[0]) # seq_len

                        # Run end to end
                        recog_decode = sess.run(dense_decode, feed_dict={input_feature_map: shared_feature_map, input_transform_matrix: transform_matrixes, input_box_mask[0]: boxes_masks, input_box_widths: box_widths})
                        recog_decode_list.extend([r for r in recog_decode])

                    timer['recog'] = time.time() - start
                    # Preparing for draw boxes
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                    if len(recog_decode_list) != boxes.shape[0]:
                        print("detection and recognition result are not equal!")
                        exit(-1)

                    scores = {}
                    score_index = 0
                    time_left = {}
                    time_index = 0
                    team_name = {}
                    quarter_dict = {}
                    remainder_attack_time = {}
                    remainder_attack_time_index = 0
                    recognition_result_num=0
                    points={}
                    for i, box in enumerate(boxes):
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                        recognition_result = ground_truth_to_word(recog_decode_list[i])

                        if contain_eng(recognition_result):
                            #print(recognition_result)
                            fix_result = bktree_search(bk_tree, recognition_result.lower())
                            #print(fix_result)
                            if len(fix_result) != 0:
                                recognition_result = fix_result[0][1]
                                #print(recognition_result)
                        else:
                            recognition_result = recognition_result

                        if recognition_result in all_team:
                            team_name[ recognition_result] = [
                                (int(box[0, 0]) + int( box[2, 0])) / 2,
                                (int(box[0, 1]) + int( box[2, 1])) / 2]
                            points[recognition_result] = [box[0, 0], box[2, 0]]

                        if recognition_result in quarter:
                            quarter_dict[ recognition_result] = [
                                (int(box[0, 0]) + int( box[2, 0])) / 2,
                                (int(box[0, 1]) + int( box[2, 1])) / 2]
                            points[recognition_result] = [box[0, 0], box[2, 0]]

                        if recognition_result.isdigit():
                            scores[ recognition_result + "_" + str(score_index)] = [
                                (int(box[0, 0]) + int( box[2, 0])) / 2,
                                (int(box[0, 1]) + int( box[2, 1])) / 2]
                            points[recognition_result + "_" + str(score_index)] = [box[0, 0], box[2, 0]]
                            score_index += 1

                        if ":" in recognition_result:
                            time_left[ recognition_result + "_" + str(time_index)] =[
                                (int(box[0, 0]) + int( box[2, 0])) / 2,
                                (int(box[0, 1]) + int( box[2, 1])) / 2]
                            points[recognition_result + "_" + str(time_index)] = [box[0, 0], box[2, 0]]
                            time_index += 1

                        if "." in recognition_result and ":" not in recognition_result:
                            remainder_attack_time[ recognition_result + "_" + str(remainder_attack_time_index)] = [
                                (int(box[0, 0]) + int( box[2, 0])) / 2,
                                (int(box[0, 1]) + int( box[2, 1])) / 2]
                            points[recognition_result + "_" + str(remainder_attack_time_index)] = [box[0, 0], box[2, 0]]
                            remainder_attack_time_index += 1

                        recognition_result_num+=1
                        str_list.append(recognition_result)
                        # Draw bounding box
                        # cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                        # Draw recognition results area
                        # text_area = box.copy()
                        # text_area[2, 1] = text_area[1, 1]
                        # text_area[3, 1] = text_area[0, 1]
                        # text_area[0, 1] = text_area[0, 1] - 15
                        # text_area[1, 1] = text_area[1, 1] - 15
                        # cv2.fillPoly(im, [text_area.astype(np.int32).reshape((-1, 1, 2))], color=(255, 255, 0))
                        # im_txt = cv2.putText(im, recognition_result, (box[0, 0], box[0, 1]), font, 0.5, (0, 0, 255), 1)
                        # 中文文字添加:
                        # im_txt = cv2ImgAddText(im, recognition_result, box[0, 0], box[0, 1], (0, 0, 149), 20)

                    if recognition_result_num == 7 or recognition_result_num == 6 or recognition_result_num == 5 or recognition_result_num==8:
                        res=get_content(remainder_attack_time,time_left,team_name, scores, quarter_dict)
                    elif recognition_result_num==9:
                        sort_points = sorted(points.items(), key=lambda item: item[1][0])
                        x_coordiate = []
                        for pair in sort_points:
                            x_coordiate.append(pair[1][0])
                            x_coordiate.append(pair[1][1])
                        x_sort = sorted(x_coordiate)
                        if x_sort == x_coordiate:
                            drop1 = sort_points[1][0]
                            drop2 = sort_points[4][0]
                            if drop1 in remainder_attack_time:
                                remainder_attack_time=remove_key(remainder_attack_time,drop1)
                            if drop2 in remainder_attack_time:
                                remainder_attack_time=remove_key(remainder_attack_time,drop2)
                            if drop1 in time_left:
                                time_left=remove_key(time_left,drop1)
                            if drop2 in time_left:
                                time_left=remove_key(time_left,drop2)
                            if drop1 in scores:
                                scores=remove_key(scores,drop1)
                            if drop2 in scores:
                                scores=remove_key(scores,drop2)
                        res = get_content(remainder_attack_time, time_left, team_name, scores, quarter_dict)
                if not FLAGS.just_infer:
                    corridate_true = corridate_list[ind].split("_")[4:]
                    label_true = label_list[ind].split("_")
                    res_true = get_score_info_v2(corridate_true, label_true)
                    if res != res_true:
                        #print(im_fn.split("/")[-1],'wrong!!!')
                        wrong += 1
                        #print(im_fn.split("/")[-1],label_list[ind],res_true,res,("_").join(str_list))
                    total += 1
                    print(im_fn.split("/")[-1], label_list[ind], res_true, res, ("_").join(str_list))
                else:
                    print(im_fn.split("/")[-1], res,("_").join(str_list))
                duration = time.time() - start_time
                #print('{} : detect {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms, recog {:.0f}ms'.format(im_fn, timer['detect']*1000, timer['restore']*1000, timer['nms']*1000, timer['recog']*1000))
            print("wrong:{}".format( wrong))
            print("total:{}".format(total))
            print("precision:{}".format((total-wrong)/total))
                # print('[timing] {}'.format(duration))


if __name__ == '__main__':
    tf.app.run()
