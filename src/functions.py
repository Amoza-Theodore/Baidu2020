import os, shutil
from ctypes import *
import cv2
import numpy as np
import time
import select
from paddlelite import *
from PIL import Image
import sys
sys.path.append('../class')

def get_image(ai_settings, car, dlmodel):
    """摄像头获取图像"""
    select.select((car.video,), (), ())
    image_data = car.video.read_and_queue()
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 获取标志物图像
    label_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    dlmodel.label_img = Image.fromarray(label_img)

    # 获取角度值图像
    lower_hsv = ai_settings.lower_hsv
    upper_hsv = ai_settings.upper_hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    angle_img = Image.fromarray(mask)
    angle_img = angle_img.resize((128, 128), Image.ANTIALIAS)
    angle_img = np.array(angle_img).astype(np.float32)
    angle_img = cv2.cvtColor(angle_img, cv2.COLOR_GRAY2BGR)
    angle_img = angle_img / 255.0;
    dlmodel.angle_img = np.expand_dims(angle_img, axis=0)

def get_follow_para(ai_settings, dlmodel, markstats):
    """得到如影随形中标志物相关的信息"""
    get_label_para(ai_settings, dlmodel, markstats)
    if markstats.detect:
        # save_img(ai_settings, dlmodel)
        print('score = {}'.format(markstats.scores))
        for label_idx, box in zip(markstats.labels, markstats.boxes):
            if ai_settings.follow_dict[label_idx] == 'landmark':
                markstats.last_follow_center_x = markstats.follow_center_x
                markstats.follow_center_x, markstats.follow_center_y = \
                    analyse_box(ai_settings, box)

def get_label_para(ai_settings, dlmodel, markstats):
    """得到标志物检测相关的信息"""
    paddle_data_feeds = dlmodel.deal_tensor()
    label_outputs = dlmodel.label_predictor.Run(paddle_data_feeds)
    label_outputs = np.array(label_outputs[0], copy=False)
    markstats.detect = check_detect(ai_settings, label_outputs)
    # 如果检测到标志物
    if markstats.detect:
        markstats.labels, markstats.scores, markstats.boxes = get_img_para(ai_settings, label_outputs)

    # 如果未检测到标志物
    elif not markstats.detect:
        pass

def analyse_box(ai_settings, box):
    """处理 box, 得到 center_x, center_y"""
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    xmin, xmax = (int(x / 608 * 320) for x in [xmin, xmax])
    ymin, ymax = (int(y / 608 * 240) for y in [ymin, ymax])
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)
    return center_x, center_y

def check_detect(ai_settings, label_outputs):
    """判断是否检测到标志物"""
    if len(label_outputs.shape) > 1:
        scores = label_outputs[:, 1]
        for score in scores:
            # 若 score > 阈值, 则表示识别成功
            if score > ai_settings.score_thresold:
                return True
    return False

def get_img_para(ai_settings, label_outputs):
    """处理 label_outputs, 得到 labels, scores, boxes"""
    mask = label_outputs[:, 1] > ai_settings.score_thresold
    labels = label_outputs[mask, 0].astype('int32')
    scores = label_outputs[mask, 1].astype('float32')
    boxes = label_outputs[mask, 2:].astype('float32')

    return labels, scores, boxes

def update_follow_para(ai_settings, car, dlmodel, markstats):
    """计算并更新跟随项目速度和角度值"""
    if markstats.detect:
        # 计算角度值并更新 [0, 320] -> [thre_left, thre_right]
        save_img(ai_settings, dlmodel)
        print('score = {}'.format(markstats.scores))
        thre_left = ai_settings.angle_thre_left
        thre_right = ai_settings.angle_thre_right
        angle = int(1500 + (160 - markstats.follow_center_x) / 320 * (thre_right - thre_left))
        # 计算速度值并更新 [0, 320*240] -> [1500, 1700]
        speed = int(1500 + (240 - markstats.follow_center_y) / 240 * 200)

        car.update(speed=speed, angle=angle)
        markstats.lose_mark_flag = True
    else:
        # 丢失标志物时, 记录当前时间
        if markstats.lose_mark_flag:
            markstats.lose_mark_flag = False
            markstats.stdtime = time.time()
        if time.time() - markstats.stdtime < 0.5 and markstats.last_follow_center_x \
                and ai_settings.remain_search_flag:
            remain_search(ai_settings, car, markstats)
        else:
            # 未检测到标志物, 小车停止运行
            car.stop()

def remain_search(ai_settings, car, markstats):
    """小车在丢失标志物时尝试自动转向搜索"""
    cnt = np.array([0, 0, 0])
    turnleft_angle = ai_settings.remain_angle_left
    turnright_angle = ai_settings.remain_angle_right
    centerx_thre_left = ai_settings.centerx_thre_left
    centerx_thre_right = ai_settings.centerx_thre_right
    if markstats.last_follow_center_x < centerx_thre_left:
        angle = turnleft_angle
        car.turn_left(angle=angle)
    elif markstats.last_follow_center_x > centerx_thre_right:
        angle = turnright_angle
        car.turn_right(angle=angle)
    else:
        car.stop()

def predict_angle(ai_settings, dlmodel):
    """获得图像所对应的角度值"""
    tmp = np.zeros((1, 128, 128, 3))
    img = dlmodel.angle_img;
    predictor = dlmodel.angle_predictor

    i = predictor.get_input(0);
    i.resize((1, 3, 128, 128));
    tmp[0, 0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
    tmp = tmp.reshape(1, 3, 128, 128);
    frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    i.set_data(tmp)

    predictor.run();
    out = predictor.get_output(0);
    score = out.data()[0][0];
    return score

def algorithmal_control(ai_settings, car, markstats, algo):
    """对不同标志物分别进行算法层面的控制"""
    dist_ctrl = ai_settings.dist_control

    # 识别标志物并更新flag
    if markstats.detect:
        identificate_mark(ai_settings, markstats, algo, dist_ctrl)

    # 执行对应的操作
    process_operation(ai_settings, car, algo)

def identificate_mark(ai_settings, markstats, algo, dist_ctrl):
    """识别标志模块"""
    for label_id, box in zip(markstats.labels, markstats.boxes):
        _, center_y = analyse_box(ai_settings, box)
        label = ai_settings.label_dict[label_id]
        if label == 'side_walk' and center_y > dist_ctrl['side_walk']:
            algo.sidewalk_flag = True
        if label == 'limit' and center_y > dist_ctrl['limit']:
            algo.limit_flag = True
        if label == 'limit_end' and center_y > dist_ctrl['limit_end']:
            algo.limit_flag = False
        if label == 'overtake' and center_y > dist_ctrl['overtake']:
            algo.overtake_flag = True

def process_operation(ai_settings, car, algo):
    """流程操作模块, 根据优先级高低排序"""
    if algo.sidewalk_flag and 'side_walk' not in algo.completed:
        car.jerk()
        time.sleep(ai_settings.sidewalk_pausetime)
        car.upwards()
        algo.completed.append('side_walk')
        algo.sidewalk_flag = False
        return

    if algo.overtake_flag:
        algo.overtake_flag = False
        algo.overtake()

    if algo.limit_flag:
        angle = ai_settings.angle_limit_formula(algo.angle_prep)
        car.update(angle=angle)
        return

    # 未进行任何操作, 直接更新速度值和角度值
    car.update(car.speed, car.angle)

def save_img(ai_settings, dlmodel):
    """保存图片"""
    output_path = os.path.join(ai_settings.img_save_path, str(dlmodel.ImgInd) + '.jpg')
    dlmodel.label_img.save(output_path)
    dlmodel.ImgInd += 1

def clean_img(ai_settings):
    """清空 predict_img 文件夹"""
    img_save_path = ai_settings.img_save_path
    if os.path.exists(img_save_path):
        shutil.rmtree(img_save_path)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)