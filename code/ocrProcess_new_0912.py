import cv2
import os
import re
import numpy as np
import time
import shutil
import sys
import binascii
import time
import shutil
import tempfile
import struct
import uuid

import cv2
import os
import re
import numpy as np
import sys
import binascii
import time
import shutil
import socket
import uuid
import pickle

DEBUG = False               #debug / no save, print image, output box information

def My_imwrite(_filename, _img, params=None):
    try:
        ext = os.path.splitext(_filename)[1]
        result, n = cv2.imencode(ext, _img, params)
        if result:
            with open(_filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False

    except Exception as e:
        print(e)
        return False



class yoloModel():
    def __init__(self, yolo_config = None):
        self.yolo_dict = {}
        self.colors = None
        self.yolo_classes = {}
        self.yolo_thresh = {}
        start_time = time.time()
        if(yolo_config is not None):
            temp_directory = 'c:/'+str(int(start_time))+'/'
            if not (os.path.isdir(temp_directory)):
                os.makedirs(os.path.join(temp_directory))

            for key, val in yolo_config.items():
                encoding_file = val[0]
                classes = []
                list_data = self.decodeFile(encoding_file)
                temp_cfg = open(temp_directory+'model.cfg','wb')
                temp_weight = open(temp_directory+'weight.weights', 'wb')
                temp_wname = open(temp_directory+'class.wname', 'wb')
                temp_cfg.write(list_data[0])
                temp_weight.write(list_data[1])
                temp_wname.write(list_data[2])
                temp_cfg.close()
                temp_weight.close()
                temp_wname.close()

                # with open(val[2]) as f:
                #     classes = " ".join([line.strip() for line in f])
                #f = open(val[2], 'r', encoding='utf-16')
                with open(temp_directory+'class.wname', 'r', encoding='utf-8') as f:
                    classes2 = f.readlines()
                for classname in classes2:
                    classes.append(classname.rstrip('\n'))
                    # for line in f:
                    #     classes.append(f.readline())
                net = cv2.dnn.readNet(temp_directory+'weight.weights', temp_directory+'model.cfg')
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

                if os.path.isfile(temp_directory+'weight.weights'):
                    os.remove(temp_directory+'weight.weights')
                if os.path.isfile(temp_directory+'model.cfg'):
                    os.remove(temp_directory+'model.cfg')
                if os.path.isfile(temp_directory+'class.wname'):
                    os.remove(temp_directory+'class.wname')


                model = cv2.dnn_DetectionModel(net)
                if key == 'SEGMENTATION':
                    model.setInputParams(size=(256,256), scale=1/256)
                elif key == 'OCR':
                    model.setInputParams(size = (64,64), scale=1/256)
                else:
                    model.setInputParams(size = (416,416), scale=1/256)

                if(net):
                    #layer_names = net.getLayerNames()
                    #output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    self.colors = np.random.uniform(0, 255, size=(len(classes), 3))
                    self.yolo_dict[key] = model
                    self.yolo_classes[key] = classes
                    self.yolo_thresh[key] = float(val[1])
            if os.path.isdir(temp_directory):
                shutil.rmtree(temp_directory)

    def getYoloModel(self):
        return self.yolo_dict

    def getClasses(self):
        return self.yolo_classes

    def getThresh(self):
        return self.yolo_thresh

    def decodeFile(self, encoding_file):
        with open(encoding_file, 'rb') as f:
            content = f.readlines()

            for i in range(0, len(content)):
                content[i] = content[i][:-1]
                content[i] = binascii.unhexlify(content[i])

            return content
        return None

#####################################
# configuration
# WAIT_SECOND : 600 - 10 minute
# NUM_MODELS : number of yolo model
# MODEL_KIND : kind of cnn
# MODEL_PATH : weight file path
# MODEL_CONF : configuration file path
# MODEL_CLASS: wname(class name) file path
############################################

class configParser():
    def __init__(self, path=os.path.dirname(os.path.realpath(__file__))):
        self.base_dir = path
        self.model_dict = {}
        self.output_folder = None
        self.input_folder = None
        self.wait_second=0
        self.grid_num = 1
        self.overlap_ratio = 0.

    def get_model_config(self):
        return self.model_dict

    def get_output_path(self):
        return self.output_folder

    def get_input_path(self):
        return self.input_folder

    def get_wait_second(self):
        return self.wait_second

    def get_grid_num(self):
        return self.grid_num

    def get_overlap_ratio(self):
        return self.overlap_ratio

    def parser(self):
        #config_path = self.base_dir + '/process.conf'
        config_path = '../process.conf'
        config_file = open(config_path, 'r')

        if(config_file):
            pattern = re.compile(r'\s+')
            #num model
            line = config_file.readline()
            num_models = line.split("|")
            #OUTPUT_FOLDER
            line = config_file.readline()
            line = re.sub(pattern, '', line)
            split_dir = line.split("|")
            self.output_folder = split_dir[1]
            #INPUT_FOLDER
            line = config_file.readline()
            line = re.sub(pattern, '', line)
            split_dir = line.split("|")
            self.input_folder = split_dir[1]
            #WAIT_SECOND
            line = config_file.readline()
            line = re.sub(pattern, '', line)
            split_dir = line.split("|")
            self.wait_second = int(split_dir[1])
            #GRID_NUM
            line = config_file.readline()
            line = re.sub(pattern, '', line)
            split_dir = line.split("|")
            self.grid_num = int(split_dir[1])
            #overlap ratio
            line = config_file.readline()
            line = re.sub(pattern,'', line)
            split_dir = line.split("|")
            self.overlap_ratio = float(split_dir[1])
            #skip line
            config_file.readline()
            if(num_models):
                model_count = 0

                while (model_count < int(num_models[1])):
                    for i in range(0, 3):
                        line = config_file.readline()
                        line = re.sub(pattern, '', line)
                        split_list = line.split("|")
                        if(split_list[0] == 'MODEL_KIND'):
                            model_kind = split_list[1]
                        elif(split_list[0] == 'MODEL_PATH'):
                            model_path = split_list[1]
                        elif(split_list[0] == 'MODEL_THRESH'):
                            model_thresh = split_list[1]
                        else:
                            return False
                    self.model_dict[model_kind] = [model_path, model_thresh]
                    print(self.model_dict[model_kind])
                    #skip line
                    line = config_file.readline()
                    model_count += 1
            config_file.close()

class ocrProcess():
    def __init__(self, config_path = os.path.dirname(os.path.realpath(__file__))):
        self.config_parser = configParser(config_path)
        self.dir_file_list = []
        self.sub_dir_list = []
        self.original_dirname = None

        self.HOST = "localhost"
        self.PORT = 4000
        #LOOP = True
        self.ID = -1
        self.mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mysocket.sendto(pickle.dumps('hi'), (self.HOST, self.PORT))
        self.packet, self.address = self.mysocket.recvfrom(1024)

        self.ID = self.packet

        #self.bFirst = False

    def __del__(self):
        self.mysocket.close()

    def sendMessage(self, message=None):
        send_data = [self.ID, message, self.HOST, self.PORT]
        send_data = pickle.dumps(send_data)
        self.mysocket.sendto(send_data, (self.HOST, self.PORT))

    def search(self, dirname, dir_file_list):
        # dir_file_list = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                #if self.bFirst == False:
                #    self.original_dirname = full_filename
                #    self.bFirst = True
                search_dir_string = full_filename.replace(self.original_dirname, "")
                self.sub_dir_list.append(search_dir_string)
                self.search(full_filename, self.dir_file_list)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.jpg' or ext == '.png' or ext == '.jpeg':
                    dir_file_list.append(full_filename)


    def printProgress(self, iteration, total, prefix='', suffix='', decimals=1, barLength=100):
        formatStr = "{0:." + str(decimals) + "f}"
        t_total = total

        percent = formatStr.format(100 * (iteration / float(t_total)))
        filledLength = int(round(barLength * iteration / float(t_total)))
        bar = '#' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
        self.sendMessage('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
        #if iteration == total-1: sys.stdout.write('\n')
        sys.stdout.flush()

    def non_max_suppression(self, boxes, probs=None, overlapThresh=0.3):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes are integers, convert them to floats -- this
        # is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # compute the area of the bounding boxes and grab the indexes to sort
        # (in the case that no probabilities are provided, simply sort on the
        # bottom-left y-coordinate)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        # if probabilities are provided, sort on them instead
        if probs is not None:
            idxs = probs

        # sort the indexes
        idxs = np.argsort(idxs)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding
            # box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        return boxes[pick].astype("int")

    def nms(self, boxes, threshold=0.3):
        '''
        :param boxes: [x,y,w,h]
        :param threshold:
        :return:
        '''
        if len(boxes) == 0:
            return []

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > threshold)[0])))
        return boxes[pick].astype("int")

    def runOCR(self, input_list_img, model, thresh):
        list_cls = []
        list_score = []
        list_boxes = []

        for charimg in input_list_img:
            character_cls, character_score, character_boxes = model.detect(charimg, thresh, 0.0)
            list_cls.extend(character_cls)
            list_score.extend(character_score)
            list_boxes.extend(character_boxes)

        return list_cls, list_score, list_boxes

    def getBinaryImg(self, input_img):
        list_img = []
        input_h, input_w = input_img.shape[:2]

        ret_1, cutimg = cv2.threshold(input_img, 90, 255, cv2.THRESH_BINARY)
        ret_2, cutimg2 = cv2.threshold(input_img, 100, 255, cv2.THRESH_BINARY)
        ret_3, cutimg3 = cv2.threshold(input_img, 150, 255, cv2.THRESH_BINARY)

        cutimg = cv2.cvtColor(cutimg, cv2.COLOR_GRAY2BGR)
        cutimg2 = cv2.cvtColor(cutimg2, cv2.COLOR_GRAY2BGR)
        cutimg3 = cv2.cvtColor(cutimg3, cv2.COLOR_GRAY2BGR)
        #cut_h, cut_w = cutimg.shape[:2]

        # cv2.imshow('img1', cutimg)
        # cv2.imshow('img2', cutimg2)
        # cv2.imshow('img3', cutimg3)
        # cv2.waitKey(0)

        rectsize = input_w
        if input_h > input_w: rectsize = input_h

        x1 = (rectsize - input_w) / 2
        y1 = (rectsize - input_h) / 2
        x2 = x1 + input_w
        y2 = y1 + input_h

        charimg = np.full((rectsize, rectsize, 3), 0, np.uint8)
        charimg[int(y1):int(y2), int(x1):int(x2)] = cutimg
        list_img.append(charimg)
        charimg = np.full((rectsize, rectsize, 3), 0, np.uint8)
        charimg[int(y1):int(y2), int(x1):int(x2)] = cutimg2
        list_img.append(charimg)
        charimg = np.full((rectsize, rectsize, 3), 0, np.uint8)
        charimg[int(y1):int(y2), int(x1):int(x2)] = cutimg3
        list_img.append(charimg)

        return list_img


    def main(self):
        self.config_parser.parser()
        print('main')
        self.sendMessage('main')

        model_config = self.config_parser.get_model_config()
        output_path = self.config_parser.get_output_path()
        input_path = self.config_parser.get_input_path()
        wait_second = self.config_parser.get_wait_second()
        _gridNum = self.config_parser.get_grid_num()        
        _overlap_ratio = self.config_parser.get_overlap_ratio()
        if(model_config is not None):
            yolo_model = yoloModel(model_config)
            dict_yolo_model = yolo_model.getYoloModel()
            yolo_classes = yolo_model.getClasses()
            yolo_thresh = yolo_model.getThresh()

            if(len(model_config) == len(dict_yolo_model)):
                print(f'success loaded {len(model_config)} model')
                self.sendMessage(f'success loaded {len(model_config)} model')
                while True:
                    self.mysocket.settimeout(1.0)
                    try:
                        packet, address = self.mysocket.recvfrom(1024)
                        if (pickle.loads(packet) == 'quit'):
                            self.mysocket.sendto(pickle.dumps('quit'), (self.HOST, self.PORT))
                            packet, address = self.mysocket.recvfrom(1024)
                            if (pickle.loads(packet) == 'quit'):
                                break
                    except socket.timeout:
                        print('timeout')

                    self.mysocket.settimeout(None)


                    self.dir_file_list = []
                    #self.search('NKLmodels/sample_images', dir_file_list)
                    self.original_dirname = input_path
                    self.search(input_path, self.dir_file_list)
                    if len(self.dir_file_list) != 0:
                        print('start process')
                        self.sendMessage('start OCR process')
                        for idx, img_path in enumerate(self.dir_file_list):
                            self.printProgress(idx+1, len(self.dir_file_list), 'OCR Processing: ', 'Complete', 1, 50)
                            while not os.path.isfile(img_path):
                                # ignore if no such file is present.
                                pass
                            
                            #tstart = datetime.now()
                            src_img = cv2.imread(img_path, 0)                   
                            src_img = 255 - src_img                            
                            input_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)     

                            input_h, input_w = input_img.shape[:2]
                            grid_size = int(input_w / _gridNum)
                            if(input_w > input_h):
                                grid_size = int(input_h / _gridNum)
                            overlap_size = int(grid_size*_overlap_ratio)
                            grid_step = grid_size - overlap_size

                            x_num = int((input_w / grid_step) + 1)
                            y_num = int((input_h / grid_step) + 1)

                            seg_cls, seg_score, seg_boxes = [],[],[]
                            for x in range(x_num):
                                for y in range(y_num):
                                    x1 = x*grid_step
                                    y1 = y*grid_step
                                    x2 = x1 + grid_size
                                    y2 = y1 + grid_size

                                    if (x2 >= input_w):
                                        x2 = input_w
                                        x1 = x2 - grid_size
                                    if (y2 >= input_h):
                                        y2 = input_h
                                        y1 = y2 - grid_size
                                    tile_img = input_img[y1:y2, x1:x2]

                                    #cv2.imshow('grid', tile_img)
                                    #cv2.waitKey(0)

                                    cls, score, boxes = dict_yolo_model['SEGMENTATION'].detect(tile_img, yolo_thresh['SEGMENTATION'], 0.1) #conf thresh, nms thresh

                                    box_area = []
                                    for index in range(len(boxes)):
                                        boxes[index][0] += x1
                                        boxes[index][1] += y1
                                        box_area.append(boxes[index][2]*boxes[index][3])
                                    #print(boxes)
                                    seg_cls.extend(cls)
                                    if(len(boxes) == 1):
                                        seg_score.extend(box_area)
                                        seg_boxes.extend(boxes)
                                    elif(len(boxes) > 1):
                                        #seg_score.extend(np.asarray(score).squeeze())
                                        seg_score.extend(np.asarray(box_area).squeeze())
                                        seg_boxes.extend(np.asarray(boxes))                                                           
       

                            np_boxes = np.array(seg_boxes)
                            np_score = np.array(seg_score)
                            nms_boxes = self.non_max_suppression(np_boxes, np_score, overlapThresh=0.5)                            
                            #seg_cls, seg_score, seg_boxes = dict_yolo_model['SEGMENTATION'].detect(input_img, yolo_thresh['SEGMENTATION'], 0.4) #conf thresh, nms thresh                           
                            
                            ocr_result= []
                            for box in nms_boxes:

                                if box[3] > 4 and box[2] > 4:
                                
                                    #coord_y = box[1]- pad if (box[1]-pad >=0) else box[1]
                                    #coord_yh = box[1] + box[3]+2*pad if (box[1]+box[3]+2*pad < input_img.shape[0]) else box[1]+box[3]
                                    #coord_x = box[0] - pad if (box[0] - pad >= 0) else box[0]
                                    #coord_xw = box[0] + box[2]+2*pad if (box[0] + box[2]+2*pad < input_img.shape[1]) else box[0]+box[2]   
                                    coord_y = box[1] + 1
                                    coord_yh = box[1] + box[3] - 2
                                    coord_x = box[0] + 1
                                    coord_xw = box[0] + box[2] -2
                                    cutimg = src_img[coord_y:coord_yh, coord_x:coord_xw]                                

                                    list_img = self.getBinaryImg(cutimg)
                                    # for img in list_img:
                                    #     cv2.imshow("img", img)
                                    #     cv2.waitKey(0)
                                    character_cls, character_score, character_boxes = self.runOCR(list_img, dict_yolo_model['OCR'], yolo_thresh['OCR'])
                                    #max score를 갖는 cls만 이용
                                    sorted_score_index = sorted(range(len(character_score)), key=character_score.__getitem__, reverse=True)

                                    if(len(character_boxes) == 0):
                                        char_classes = yolo_classes['OCR']
                                        #char_color = yolo_model.colors[0]
                                        char_label = "%s %d %d %d %d %f" % (char_classes[0], int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]), 0)
                                        ocr_result.append(char_label)

                                        unique_filename = "Errors/" + str(uuid.uuid4()) + ".jpg"
                                        My_imwrite(unique_filename, cutimg)
                                        #cv2.rectangle(input_img, box, (0, 0, 255), 1)
                                        #cv2.imshow(unique_filename, charimg)
                                        #cv2.waitKey(0)
                                    else:
                                        char_classes = yolo_classes['OCR']
                                        char_color = yolo_model.colors[int(character_cls[sorted_score_index[0]]) % len(yolo_model.colors)]
                                        char_label = "%s %d %d %d %d %f" % (char_classes[int(character_cls[sorted_score_index[0]])], int(box[0]), int(box[1]),
                                        int(box[0] + box[2]), int(box[1] + box[3]), character_score[sorted_score_index[0]] * 100)
                                        if DEBUG:
                                            print(char_label)
                                            cv2.rectangle(input_img, box, (0, 255, 0), 1)
                                        else:
                                            ocr_result.append(char_label)

                                    # for (char_classid, char_score, char_box) in zip(character_cls, character_score, character_boxes):
                                    #     char_classes = yolo_classes['OCR']
                                    #     char_color = yolo_model.colors[int(char_classid[0]) % len(yolo_model.colors)]
                                    #     char_label = "%s %d %d %d %d %f" % (char_classes[int(char_classid[0])], int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]), char_score[0]*100)
                                    #     #cv2.rectangle(character, char_box, char_color, 1)
                                    #     if DEBUG:
                                    #         print(char_label)
                                    #         cv2.rectangle(input_img, box, (0, 255, 0), 1)
                                    #     else:
                                    #         ocr_result.append(char_label)
                                        #cv2.imshow('ocr result', character)
                                        #cv2.waitKey(0)

                                #cv2.rectangle(input_img, box, color, 1)
                                #cv2.waitKey(0)
                                #cv2.putText(input_img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            if DEBUG:
                                # for box in seg_boxes:                               
                                cv2.imshow('input_img', input_img)
                                cv2.waitKey(0)
                            else:
                                if not (os.path.isdir(output_path)):
                                    os.makedirs(os.path.join(output_path))
                                for output_save_path in reversed(self.sub_dir_list):
                                    if not (os.path.isdir(output_path + '/' + output_save_path)):
                                        os.makedirs(os.path.join(output_path + '/' + output_save_path))
                                # if not (os.path.isdir(output_path+'/img/')):
                                #    os.makedirs(os.path.join(output_path+'/img/'))
                                split_path = img_path.replace(self.original_dirname, "")
                                # split_path = os.path.basename(img_path)
                                shutil.move(img_path, output_path + '/' + split_path)

                                file_name = os.path.basename(img_path)
                                test_dir = split_path.replace(file_name, "")
                                text_name = os.path.splitext(file_name)
                                f = open(output_path +'/'+ test_dir + '/' + text_name[0] + '.txt', 'w')
                                for text in ocr_result:
                                    f.writelines(text)
                                    f.write('\n')
                                f.close()


                            #tend = datetime.now()                            
                            #print((tend - tstart).total_seconds()*1000)

                    else:
                        #print(' ')
                        #for idx, img_path in enumerate(dir_file_list):
                        #    print(img_path)

                        print('waiting...')
                        self.sendMessage('waiting...')
                        for i in range(0, wait_second):
                            self.mysocket.settimeout(1.0)
                            try:
                                packet, address = self.mysocket.recvfrom(1024)
                                if (pickle.loads(packet) == 'quit'):
                                    self.mysocket.sendto(pickle.dumps('quit'), (self.HOST, self.PORT))
                                    packet, address = self.mysocket.recvfrom(1024)
                                    if (pickle.loads(packet) == 'quit'):
                                        break
                            except socket.timeout:
                                print('timeout')

                            self.mysocket.settimeout(None)
                            time.sleep(1.0)

if __name__ == '__main__':
    ocr_process = ocrProcess()
    ocr_process.main()


    # print(ID, packet, address)
    # while LOOP:



        #
        #time.sleep(10)
        #print(ID, packet, address)
        #if(ID == packet):
        #    break
        #time.sleep(10)




