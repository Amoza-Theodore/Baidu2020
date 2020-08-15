import os, shutil
import v4l2capture
from ctypes import *
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import getopt
import sys, select, termios, tty
import threading
import paddlemobile as pm
from paddlelite import *
import codecs
import multiprocessing
import math
import functools
from PIL import Image
from PIL import ImageFile
from PIL import ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Car:
    def __init__(self):
        # load config
        self.config()

        # create folders
        if not os.path.exists(self.img_save_path): os.makedirs(self.img_save_path)
        if not os.path.exists(self.data_collect_path): os.makedirs(self.data_collect_path)
        img_collect_path = os.path.join(self.data_collect_path, 'img')
        if not os.path.exists(img_collect_path): os.makedirs(img_collect_path)

        # Initialize the camera
        camera = "/dev/video2"
        video = v4l2capture.Video_device(camera)
        video.set_format(424, 240, fourcc='MJPG')
        video.create_buffers(1)
        video.queue_all_buffers()
        video.start()
        self.video = video

        # Initialize the predictor
        self.angle_predictor = self.load_angle_model();
        self.label_predictor = self.load_label_model()

        # Initialize the lower machine
        path = os.path.split(os.path.realpath(__file__))[0] + "/.."
        lib_path = path + "/lib" + "/libart_driver.so"
        so = cdll.LoadLibrary
        self.lib = so(lib_path)
        car = "/dev/ttyUSB0"
        self.lib.art_racecar_init(38400, car.encode("utf-8"))

    def clean_predict_img(self):
        filepath = self.img_save_path
        if os.path.exists(filepath): shutil.rmtree(filepath)
        os.makedirs(filepath)

    def clean_data_collect(self):
        filepath = self.data_collect_path
        if os.path.exists(filepath): shutil.rmtree(filepath)
        os.makedirs(filepath)
        os.makedirs(filepath + '/img')

    def getvalue(self):
        axis_states = {}
        button_states = {}

        axis_map = []
        button_map = []

        buf = array.array('u', str(['\0'] * 5))
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)
        js_name = buf.tostring()

        # get number of axes and buttons
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf)  # JSIOCGAXES
        num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
        num_buttons = buf[0]

        # Get the axis map
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf)  # JSIOCGAXMAP
        for axis in buf[:num_axes]:
            axis_name = self.axis_names.get(axis, 'unknow(0x%02x)' % axis)
            axis_map.append(axis_name)
            axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf)  # JSIOCGBTNMAP

        for btn in buf[:num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            button_map.append(btn_name)
            button_states[btn_name] = 0

        return axis_map, axis_states, button_map, button_states

    def config(self):
        # load config, modifiable
        #   Where angle_model is stored
        self.angle_model_path = '../model/angle_model/model_infer'
        #   Where label_model is stored
        self.label_model_path = '../model/label_model/model_infer'
        #   Where images are saved
        self.img_save_path = '../predict_img'
        #   Where the collected data is saved
        self.data_collect_path = '../data_collect'
        #   The speed at which the car runs
        self.init_vels = 1600
        #   The corresponding serial number and label
        self.label_dict = {
            0: 'green_light',
            1: 'limit',
            2: 'limit_end',
            3: 'outer',
            4: 'red_light',
            5: 'stop',
            6: 'straight',
            7: 'turn_left'
        }
        #   The sequence number required to save the image
        self.ImgInd = 0
        #   Flags for marker detection
        self.stop_flag = False
        self.run_flag = True
        self.limit_flag = False
        self.turn_left_flag = False
        self.P_flag = False
        self.PR_flag = False

    def dataset(self, video):
        lower_hsv = np.array([25, 75, 190])
        upper_hsv = np.array([40, 255, 255])

        select.select((video,), (), ())
        image_data = video.read_and_queue()
        frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        img_angle = Image.fromarray(mask)
        img_angle = img_angle.resize((128, 128), Image.ANTIALIAS)
        img_angle = np.array(img_angle).astype(np.float32)
        img_angle = cv2.cvtColor(img_angle, cv2.COLOR_GRAY2BGR)
        img_angle = img_angle / 255.0;
        img_angle = np.expand_dims(img_angle, axis=0)

        img_label = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img_label = Image.fromarray(img_label)
        return img_label, img_angle, img_save;

    def load_angle_model(self):
        valid_places = (
            Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
            Place(TargetType.kHost, PrecisionType.kFloat),
            Place(TargetType.kARM, PrecisionType.kFloat),
        );
        config = CxxConfig();
        model_dir = self.angle_model_path;
        config.set_model_file(model_dir + "/model");
        config.set_param_file(model_dir + "/params");
        config.set_valid_places(valid_places);
        predictor = CreatePaddlePredictor(config);
        return predictor;

    def load_label_model(self):
        model_dir = self.label_model_path
        pm_config = pm.PaddleMobileConfig()
        pm_config.precision = pm.PaddleMobileConfig.Precision.FP32
        pm_config.device = pm.PaddleMobileConfig.Device.kFPGA
        pm_config.model_dir = model_dir
        pm_config.thread_num = 4
        label_predictor = pm.CreatePaddlePredictor(pm_config)

        return label_predictor

    def tensor_deal(self, origin):
        tensor_img = origin.resize((256, 256), Image.BILINEAR)
        if tensor_img.mode != 'RGB':
            tensor_img = tensor_img.convert('RGB')
        tensor_img = np.array(tensor_img).astype('float32').transpose((2, 0, 1))
        tensor_img -= 127.5
        tensor_img *= 0.007843
        tensor_img = tensor_img[np.newaxis, :]
        tensor = pm.PaddleTensor()
        tensor.dtype = pm.PaddleDType.FLOAT32
        tensor.shape = (1, 3, 256, 256)
        tensor.data = pm.PaddleBuf(tensor_img)
        paddle_data_feeds = [tensor]
        return paddle_data_feeds

    def angle_predict(self, predictor, image):
        tmp = np.zeros((1, 128, 128, 3))
        img = image;

        i = predictor.get_input(0);
        i.resize((1, 3, 128, 128));
        tmp[0, 0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
        tmp = tmp.reshape(1, 3, 128, 128);
        frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        i.set_data(tmp)

        predictor.run();
        out = predictor.get_output(0);
        score = out.data()[0][0];
        return score;

    def get_img_para(self, label_outputs):
        # If the score > 0.5 then the object is detected successfully
        # Test: hyber-para
        mask = label_outputs[:, 1] > 0.6 if len(label_outputs.shape) > 1 else None
        if mask is not None:
            detect = True
            labels = label_outputs[mask, 0].astype('int32')
            scores = label_outputs[mask, 1].astype('float32')
            boxes = label_outputs[mask, 2:].astype('float32')

        # No objects were detected
        else:
            detect = False
            labels = None
            scores = None
            boxes = None

        return detect, labels, scores, boxes

    def img_save(self, img, detect, boxes, labels, scores):
        img = Image.fromarray(img)

        # # Detect object, draw a rectangle around the picture
        # if detect == True:
        #     # draw = ImageDraw.Draw(img)
        #     for box, label, score in zip(boxes, labels, scores):
        #         xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        #         xmin, xmax = (int(x / 608 * 320) for x in [xmin, xmax])
        #         ymin, ymax = (int(y / 608 * 240) for y in [ymin, ymax])
        #
        #         draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')
        #         box_str = str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
        #         draw.text((xmin, ymin), self.label_dict[int(label)] + ' ' + str(score) + '\n' + box_str, (255, 255, 0))

        # save image
        output_path = os.path.join(self.img_save_path, str(self.ImgInd) + '.jpg')
        img.save(output_path)
        self.ImgInd += 1

    def user_cmd(self, detect, label_ids, scores, boxes, vel, angle, a):
        # identify
        if detect:
            for label_id, box in zip(label_ids, boxes):
                # deal box
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                xmin, xmax = (int(x / 608 * 320) for x in [xmin, xmax])
                ymin, ymax = (int(y / 608 * 240) for y in [ymin, ymax])
                center_y = int((ymin + ymax) / 2)
                label = self.label_dict[label_id]
                print('label: ' + label)
                # if center_y > 160:
                #     print('label = ' + label)
                if label == 'stop':
                    if center_y > 60:
                        print('label = ' + label)
                        self.lib.send_cmd(1500, 1500)
                        self.run_flag = False
                if label == 'red_light':
                    if center_y >35:
                        print('label ='+label)
                        self.lib.send_cmd(1500, 1500)


                        self.stop_flag = True
                if label == 'green_light':
                    if center_y > 25:
                        print('label = ' + label)
                        self.stop_flag = False
                if label == 'limit':
                    if center_y > 120:
                        print('label = ' + label)
                        self.limit_flag = True
                if label == 'limit_end':
#                    if 120<center_y <200:
#                        self.lib.send_cmd(15,2100)  
                    if center_y >200:            
                        print(center_y)
                        print('label = ' + label)
                        self.lib.send_cmd(1528,1760)
                        time.sleep(0.4)
                        self.lib.send_cmd(1528,1550)
                        time.sleep(0.1)
                        self.limit_flag = False
                        self.P_flag=True          
                        # self.lib.send_cmd(vel, 2000)


                if label == 'turn_left':
                    if center_y > 150:
                        print('label = ' + label)
                        # self.std_time = time.time()
                        self.turn_left_flag = True
                        self.P_flag = False
                        self.PR_flag = False
                if label == 'straight':
                    if center_y > 160:
                        print('label = ' + label)
                        self.PR_flag = True
                        
                        self.PT_flag = True               
                        pass

        # operation
        if self.stop_flag:
            self.lib.send_cmd(1500, 1500)
            return

        if self.turn_left_flag:
            time.sleep(0.6)
            self.lib.send_cmd(vel,2250)
            time.sleep(0.3)
            self.turn_left_flag=False
            # nowtime = time.time()
            # if nowtime - self.std_time > 1.2:
            #     self.lib.send_cmd(vel, angle)
            #     self.turn_left_flag = False
            #     return
            # if nowtime - self.std_time < 0.68:
            #     self.lib.send_cmd(vel, angle)
            #     return
            #     self.lib.send_cmd(vel, 2100)
            return

        if self.limit_flag:
            # Test: unfinished
            print('limited speed')
#            angle = int(-2174 * a * a + 3805 * a + 141.3)
#            angle = int(-2083 * a * a + 3695* a +151.4)
#            angle = int(-1768* a * a +3398 * a + 177.9)
            angle = int(-1792* a * a +3413* a + 176.7+2)
            print(angle)
            self.lib.send_cmd(1528, angle)
            return
            
        if self.P_flag & self.PR_flag:
            vel = 1600
            if self.PT_flag :
                self.lib.send_cmd(1600, 915)
                time.sleep(0.1)
                self.PT_flag = False
             
#            angle = int(-3132*a*a*a+5114*a*a-522.2*a+874.4+30+20)
            angle= int(-2635*a*a*a+4710*a*a-528.6*a+921.3-80)
#            a= int(-2635*angle*angle*angle+4710*angle*angle-528.6*angle+921.3-38)  #test  4  8.27  右转
            print("  turn  right")
               

        # normal situation
        print('angle = ' + str(angle))
        self.lib.send_cmd(vel, angle)
        return

            #-----------------------operation module-----------------------#

    #-------------------------------------------- operation system --------------------------------------------#

    # The main program of Lane Identify
    def run_lane(self):
        while True:
            # Access to images, img_label, img_angle, and img_save are the files required by label_predictor,angle_predictor, and img_save
            img_label, img_angle, img_save = self.dataset(self.video)

            # Get vel
            vel = self.init_vels

            # Predict angle, the result interval is [800, 2100]
            angle = self.angle_predict(self.angle_predictor, img_angle)
            angle = int(angle * 1570 + 740)

            self.lib.send_cmd(vel, angle)

    # The main program of Lane Identify and Label Identify
    def run(self):
        while self.run_flag:
            # Access to images, img_label, img_angle, and img_save are the files required by label_predictor,angle_predictor, and img_save
            img_label, img_angle, img_save = self.dataset(self.video)

            # Transform the img_label image into tensor
            paddle_data_feeds = self.tensor_deal(img_label)

            # Get vel
            # vel = self.init_vels
            # Test
            vel = 1600

            # Predict angle, the result interval is [800, 2100]
            a = self.angle_predict(self.angle_predictor, img_angle)
            # angle = int(angle * 1570 + 740)
            # Test
            angle = int(-3132*a*a*a+5114*a*a-522.2*a+874.4)
#            angle = int(-1810*a*a*a+4083*a*a-558.3*a+899)

#            angle = int(-2223*a*a*a+4129*a*a-296.1*a+898.3) #test  3
            #angle = int(-654.8 * angle * angle + 3018 * angle + 160.8)
            if a > 0.65:
                angle = int(-3132*a*a*a+5114*a*a-522.2*a+874.4+5)

            elif 0.1<a <0.45:
            
                
                angle = int(-3132*a*a*a+5114*a*a-522.2*a+874.4)
              
            elif 0.45<a<0.65:
                angle= int(-2635*a*a*a+4710*a*a-528.6*a+921.3-180)
            elif a <0.1:
                angle = 500
                
            print(a)
#            angle= int(-2635*a*a*a+4710*a*a-528.6*a+921.3-38-45)


            # Predict label, the results are None(no object) or labels, scores, boxes
            label_outputs = self.label_predictor.Run(paddle_data_feeds)
            label_outputs = np.array(label_outputs[0], copy=False)

            # Get picture parameters
            detect, labels, scores, boxes = self.get_img_para(label_outputs)

            # save img
            # self.img_save(img=img_save, detect=detect, boxes=boxes, labels=labels, scores=scores)

            # Sends data to the control program
            self.user_cmd(detect, labels, scores, boxes, vel, angle,a)

        if not self.run_flag:
            self.lib.send_cmd(1500, 1500)

if __name__ == '__main__':
    car = Car()
    car.run()