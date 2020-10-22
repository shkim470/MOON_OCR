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
import nms
import tempfile
import struct
import uuid

import socket
import pickle
import scipy.spatial.distance as distance

DEBUG = False               #debug / no save, print image, output box information
ONLY_PROCESS = True

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
        self.dict_yolo_model = []
        self.yolo_classes = []
        self.yolo_thresh = []

        self.original_folder_name = ''
        self.working_folder_name = ''



        if ONLY_PROCESS == True:
            self.HOST = "localhost"
            self.PORT = 4000
            #LOOP = True
            self.ID = -1
            self.mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.mysocket.sendto(pickle.dumps('hi'), (self.HOST, self.PORT))
            self.packet, self.address = self.mysocket.recvfrom(1024)

            self.ID = self.packet
        else:
            self.mysocket = None
            self.ID = None
            self.HOST = None
            self.PORT = None
    
    def __del__(self):
        if self.mysocket is not None:
            self.mysocket.close()
            restoreFolderName()
            print("distruction------")

    def restoreFolderName():
        if os.path.exists(self.working_folder_name) is True and  os.path.exists(self.original_folder_name) is False:
            os.rename(self.working_folder_name, self.original_folder_name)


    
    def sendMessage(self, message=None):
        if ONLY_PROCESS == True:
            send_data = [self.ID, message, self.HOST, self.PORT]
            send_data = pickle.dumps(send_data)
            if self.mysocket is not None:
                self.mysocket.sendto(send_data, (self.HOST, self.PORT))

    def search(self, dirname, dir_file_list):
        #dir_file_list = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                self.search(full_filename, dir_file_list)
            else:
                #ext = os.path.splitext(full_filename)[-1]
                #if ext == '.jpg' or ext == '.png' or ext == '.jpeg':
                dir_file_list.append(full_filename)


    def printProgress(self, iteration, total, prefix='', suffix='', decimals=1, barLength=100):
        formatStr = "{0:." + str(decimals) + "f}"
        t_total = total

        percent = formatStr.format(100 * (iteration / float(t_total)))
        filledLength = int(round(barLength * iteration / float(t_total)))
        #bar = '#' * filledLength + '-' * (barLength - filledLength)
        #sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
        #if iteration == total-1: sys.stdout.write('\n')
        if ONLY_PROCESS == True:
        #    self.sendMessage('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
            self.sendMessage('\r%s %s%s %s' % (prefix, percent, '%', suffix))
        else:
            bar = '#' * filledLength + '-' * (barLength - filledLength)
            sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
            if iteration == total-1: sys.stdout.write('\n')
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


#-------------------------------------------------------------------------------------
    def ocr_page(self, _input_img, _input_h, _input_w, _x_num, _y_num, _grid_size, _grid_step, ocr_results, ocr_result_list):

        src_img = cv2.cvtColor(_input_img, cv2.COLOR_BGR2GRAY)
        seg_cls, seg_score, seg_boxes = [],[],[]
        for x in range(_x_num):
            for y in range(_y_num):  
                x1 = x*_grid_step
                y1 = y*_grid_step   
                x2 = x1 + _grid_size
                y2 = y1 + _grid_size

                if (x2 >= _input_w): x2 = _input_w;   x1 = x2 - _grid_size
                if (y2 >= _input_h): y2 = _input_h;   y1 = y2 - _grid_size
                tile_img = _input_img[y1:y2, x1:x2]

                #inference code - detect characters
                clsid, score, boxes = self.dict_yolo_model['SEGMENTATION'].detect(tile_img, self.yolo_thresh['SEGMENTATION'], 0.1) #conf thresh, nms thresh

                
                box_area = []
                for index in range(len(boxes)): 
                    boxes[index][0] += x1;  
                    boxes[index][1] += y1;   
                    box_area.append(boxes[index][2]*boxes[index][3])

                seg_cls.extend(clsid)
                if(len(boxes) == 1):    
                    seg_score.extend(box_area) 
                    seg_boxes.extend(boxes)
                elif(len(boxes) > 1):   
                    seg_score.extend(np.asarray(box_area).squeeze())
                    seg_boxes.extend(np.asarray(boxes))        
                                
        np_boxes = np.array(seg_boxes)
        np_score = np.array(seg_score)
        nms_boxes = self.non_max_suppression(np_boxes, np_score, overlapThresh=0.5)       # check duplication and clean up                                

        for box in nms_boxes:
            if box[3] > 4 and box[2] > 4: 
                coord_y = box[1];   coord_yh = box[1] + box[3];     coord_x = box[0];   coord_xw = box[0] + box[2]
                cutimg = src_img[coord_y:coord_yh, coord_x:coord_xw]  
                cut_h, cut_w = cutimg.shape[:2]

                rectsize = cut_w
                if cut_h > cut_w: rectsize = cut_h
                x1 = (rectsize - cut_w) / 2;    y1 = (rectsize - cut_h) / 2;    x2 = x1 + cut_w; y2 = y1 + cut_h

                charimg = np.full((rectsize, rectsize), 0, np.uint8)
                charimg[int(y1):int(y2), int(x1):int(x2)] = cutimg     
                #charimg = cv2.resize(charimg,dsize=(128, 128), interpolation=cv2.INTER_AREA)  


                ret, charimg3 = cv2.threshold(charimg, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)                
                #ret, charimg3 = cv2.threshold(charimg, 100,255, cv2.THRESH_BINARY)                    
                charimg3 = cv2.cvtColor(charimg3, cv2.COLOR_GRAY2BGR)  
                character_cls3, character_score3, character_boxes3 = self.dict_yolo_model['OCR'].detect(charimg3, self.yolo_thresh['OCR'], 0.0)

                char_classid = 0; char_score = 0
                for (c, s, b) in zip(character_cls3, character_score3, character_boxes3):
                    if s > char_score:  
                        char_classid = c;   
                        char_score = s
                
                # recoinize with different options---------------------------------                
                if char_score < 0.8:
                    #ret, charimg1 = cv2.threshold(charimg, 100,255, cv2.THRESH_BINARY)
                    ret, charimg1 = cv2.threshold(charimg, 100,255, cv2.THRESH_BINARY)
                    charimg1 = cv2.cvtColor(charimg1, cv2.COLOR_GRAY2BGR) 
                    character_cls1, character_score1, character_boxes1 = self.dict_yolo_model['OCR'].detect(charimg1, self.yolo_thresh['OCR'], 0.0)
                    for (c, s, b) in zip(character_cls1, character_score1, character_boxes1):
                        if s > char_score:  
                            char_classid = c;   
                            char_score = s     
                    
                    if char_score < 0.1:
                        charimg2 = cv2.resize(charimg,dsize=(128, 128), interpolation=cv2.INTER_AREA)
                        #ret, charimg2 = cv2.threshold(charimg2, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        ret, charimg2 = cv2.threshold(charimg2, 127,255, cv2.THRESH_BINARY)

                        charimg2 = cv2.cvtColor(charimg2, cv2.COLOR_GRAY2BGR) 
                        character_cls2, character_score2, character_boxes2 = self.dict_yolo_model['OCR'].detect(charimg2, self.yolo_thresh['OCR'], 0.0)                        
                        for (c, s, b) in zip(character_cls2, character_score2, character_boxes2):
                            if s > char_score:  
                                char_classid = c;   
                                char_score = s   
                                            
                                      
                #--------------------------------------------------------------------                
                if(char_score == 0):
                    zero_score = np.array([0.50001])
                    zero_score = zero_score.astype(np.float32)
                    char_classes = self.yolo_classes['OCR']                    
                    char_label = "%s %d %d %d %d %f" % (char_classes[0], int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]), zero_score*100)
                    ocr_result_tup = [char_classes[0], [[int(box[0]), int(box[1])], [int(box[0]+box[2]), int(box[1])], [int(box[0]+ box[2]), int(box[1]+box[3])], [int(box[0]), int(box[1]+box[3])]], zero_score*100]                    
                    ocr_result_list.append(ocr_result_tup)
                    ocr_results.append(char_label)   

                    if DEBUG:
                        unique_filename = "Errors/" + str(uuid.uuid4()) + ".jpg"
                        My_imwrite(unique_filename, charimg1)    
                else:
                    char_classes = self.yolo_classes['OCR']
                    #char_color = yolo_model.colors[int(char_classid) % len(yolo_model.colors)]
                    char_label = "%s %d %d %d %d %f" % (char_classes[int(char_classid)], int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]), char_score*100)                    
                    ocr_result_tup = [char_classes[int(char_classid)], [[int(box[0]), int(box[1])], [int(box[0]+box[2]), int(box[1])], [int(box[0]+ box[2]), int(box[1]+box[3])], [int(box[0]), int(box[1]+box[3])]], char_score * 100]

                    ocr_result_list.append(ocr_result_tup)
                    ocr_results.append(char_label)   

                if DEBUG:
                    print(char_label)
                    cv2.rectangle(_input_img, box, (0, 255, 0), 1)

                    
            
        

#-----------------------------------------------------------------------------------------
    def proc_imagefile(self,_img_path, _folder_name):
        #if ONLY_PROCESS == False:   
        #    start = time.time()
        _gridNum = self.config_parser.get_grid_num()  
        _overlap_ratio = self.config_parser.get_overlap_ratio() 
        output_path = self.config_parser.get_output_path()

        src_img = cv2.imread(_img_path, 0)
        if src_img is None:            
            if not(os.path.isdir(output_path)): 
                os.makedirs(os.path.join(output_path))
            if not (os.path.isdir(output_path+'/'+_folder_name+'/tmp/')): 
                os.makedirs(os.path.join(output_path+'/'+_folder_name+'/tmp/'))
            split_path = os.path.basename(_img_path)
            shutil.move(_img_path, output_path+'/'+_folder_name+'/tmp/' + split_path)
        else:
            src_img = 255 - src_img    
            input_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)  
            
            input_h, input_w = input_img.shape[:2]        
            grid_size = int(input_w / _gridNum)
            if(input_w > input_h):  grid_size = int(input_h / _gridNum)

            overlap_size = int(grid_size*_overlap_ratio)
            grid_step = grid_size - overlap_size
            x_num = int((input_w / grid_step) + 1)
            y_num = int((input_h / grid_step) + 1)

            ocr_result= []
            ocr_result_list = [] 
            self.ocr_page(input_img, input_h, input_w, x_num, y_num, grid_size, grid_step, ocr_result, ocr_result_list)
            
            
            #sort ocr results
            sorted_list, sorted_coord_list, sorted_prob_list, sorted_coord_rb_list = self.sortingBoundingBox(ocr_result_list)
            ocr_result = []
            
            for labels, coords, probs, rbs in zip(sorted_list, sorted_coord_list, sorted_prob_list, sorted_coord_rb_list):
                for i in range(1, len(labels)):
                    first_coord = rbs[i - 1][0]
                    next_coord = coords[i][0]
                    box_height = abs(coords[i - 1][1] - rbs[i - 1][1])
                    char_label = "%s %d %d %d %d %f" % (labels[i-1], coords[i-1][0], coords[i-1][1], rbs[i-1][0], rbs[i-1][1], probs[i-1])
                    #print(char_label)
                    #Space
                    if abs(next_coord - first_coord) > box_height / 2.:
                        space = "%s %d %d %d %d %f" % (' ', coords[i-1][0], coords[i-1][1], rbs[i-1][0], rbs[i-1][1], 1.0)
                        ocr_result.append(char_label)
                        ocr_result.append(space)
                        #print(labels[i - 1], ' ')
                        #print('띄어쓰기!')
                    else:
                        ocr_result.append(char_label)
                        #print(labels[i - 1], ' ')

                if (len(labels) != 0):
                    char_label = "%s %d %d %d %d %f" % (labels[len(labels) - 1], coords[len(labels) - 1][0], coords[len(labels) - 1][1], rbs[len(labels) - 1][0], rbs[len(labels) - 1][1], probs[len(labels) - 1])
                    ocr_result.append(char_label)
                
                enter = "%s %d %d %d %d %f" % ('/n', coords[len(labels) - 1][0], coords[len(labels) - 1][1], rbs[len(labels) - 1][0], rbs[len(labels) - 1][1], 1.0)
                ocr_result.append(enter)
                #print('줄바꿈') 
            
            
            if DEBUG:
                # for box in seg_boxes:                               
                cv2.imshow('input_img', input_img)
                cv2.waitKey(0)
            else:
                if not(os.path.isdir(output_path)): os.makedirs(os.path.join(output_path))
                if not (os.path.isdir(output_path+'/'+_folder_name+'/')): os.makedirs(os.path.join(output_path+'/'+_folder_name+'/'))
                split_path = os.path.basename(_img_path)
                shutil.move(_img_path, output_path+'/'+_folder_name+'/' + split_path)
                #shutil.move(img_path, output_path+ '/' + split_path)

                text_name = os.path.splitext(split_path)
                f = open(output_path+'/'+_folder_name+'/'+text_name[0]+'.txt', 'w', encoding='utf-8')
                for text in ocr_result:
                    f.writelines(text)
                    f.write('\n')
                f.close()

        #if ONLY_PROCESS == False:                   
            #end = time.time()
            #elapsed = end - start
            #print('  ---  Elapsed time is %f seconds.' % elapsed)
#----------------------------------------------------------------------------------------------------
    def procssing_folder(self, _folder_name):
        start = time.time()
        self.original_folder_name = _folder_name
        self.working_folder_name = _folder_name


        strflag = _folder_name[0:5]
        folder_path = "{}/{}/".format(self.config_parser.get_input_path(), _folder_name)
        if os.path.exists(folder_path):
            if strflag != '_____':                
                new_folder_path = "{}/_____{}/".format(self.config_parser.get_input_path(), _folder_name)
                self.working_folder_name = new_folder_path
                os.rename(folder_path, new_folder_path)

                if os.path.exists(new_folder_path):
                    dir_file_list = []            
                    self.search(new_folder_path, dir_file_list)
                    #dir_file_list = os.listdir(new_folder_path)

                    if len(dir_file_list) != 0:
                        print(' ')
                        print('start processing---{}'.format(_folder_name))
                        if ONLY_PROCESS == True and self.mysocket is not None:
                            self.sendMessage('start processing---{}'.format(_folder_name))

                        page_num = len(dir_file_list)
                        for idx, img_path in enumerate(dir_file_list):                            
                            while not os.path.isfile(img_path):                                
                                pass
                            #processing image file in the folder                            
                            self.proc_imagefile(img_path, _folder_name)        

                            end = time.time()      
                            elapsed = '%5.2f s/file' % ((end-start)/(idx+1))
                            self.printProgress(idx+1, page_num, 'OCR({}): '.format(_folder_name), '완료 ({}/{}) 처리속도:{}'.format((idx+1), page_num, elapsed), 1, 50)
                
                    #remove the folder where all files have been processed          
                    #os.rmdir(new_folder_path)
                    shutil.rmtree(new_folder_path, ignore_errors=True)
        #end = time.time()
        #elapsed = end - start
        #print('  ---  Elapsed time is %f seconds.' % elapsed)

    def sortingBoundingBox(self, points):

        points = list(map(lambda x: [x[0], x[1][0], x[1][2], x[2], x[1][2]], points))
        # print(points)
        points_sum = list(map(lambda x: [x[0], x[1], sum([x[1][0],x[1][1]]), x[2][1], x[3], x[4]], points))
        x_y_cordinate = list(map(lambda x: x[1], points_sum))
        probability = list(map(lambda x: x[4], points_sum))
        x_y_cordinate_rb = list(map(lambda x : x[5], points_sum))
        final_sorted_list = []
        final_coord_list = []
        final_prob_list = []
        final_coord_list_rb = []
        while True:
            try:
                if len(points_sum) == 0:
                    break
                new_sorted_text = []
                new_sorted_coord = []
                new_sorted_prob = []
                new_sorted_rb = []
                initial_value_A = [i for i in sorted(enumerate(points_sum), key=lambda x: x[1][2])][0]
                #         print(initial_value_A)
                threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
                threshold_value = (threshold_value / 2) + 5
                del points_sum[initial_value_A[0]]
                del x_y_cordinate[initial_value_A[0]]
                del probability[initial_value_A[0]]
                del x_y_cordinate_rb[initial_value_A[0]]
                #         print(threshold_value)
                A = [initial_value_A[1][1]]
                K = list(map(lambda x: [x, abs(x[1] - initial_value_A[1][1][1])], x_y_cordinate))
                K = [[count, i] for count, i in enumerate(K)]
                K = [i for i in K if i[1][1] <= threshold_value]
                sorted_K = list(map(lambda x: [x[0], x[1][0]], sorted(K, key=lambda x: x[1][1])))
                B = []
                points_index = []
                for tmp_K in sorted_K:
                    points_index.append(tmp_K[0])
                    B.append(tmp_K[1])
                if len(B) == 0:
                    continue
                dist = distance.cdist(A, B)[0]
                d_index = [i for i in sorted(zip(dist, points_index), key=lambda x: x[0])]
                new_sorted_text.append(initial_value_A[1][0])
                new_sorted_coord.append(initial_value_A[1][1])
                new_sorted_prob.append(initial_value_A[1][4])
                new_sorted_rb.append(initial_value_A[1][5])

                index = []
                for j in d_index:
                    new_sorted_text.append(points_sum[j[1]][0])
                    new_sorted_coord.append(x_y_cordinate[j[1]])
                    new_sorted_prob.append(probability[j[1]])
                    new_sorted_rb.append(x_y_cordinate_rb[j[1]])
                    index.append(j[1])
                for n in sorted(index, reverse=True):
                    del points_sum[n]
                    del x_y_cordinate[n]
                    del probability[n]
                    del x_y_cordinate_rb[n]
                final_sorted_list.append(new_sorted_text)
                final_coord_list.append(new_sorted_coord)
                final_prob_list.append(new_sorted_prob)
                final_coord_list_rb.append(new_sorted_rb)
                #print(new_sorted_text)
            except Exception as e:
                print(e)
                break

        return final_sorted_list, final_coord_list, final_prob_list, final_coord_list_rb
#--------------------------------------------------------------------------------------------------------
    def main(self):
        print('main')        
        if ONLY_PROCESS == True:
            self.sendMessage('Ready...')
        
        self.config_parser.parser()
        

        model_config = self.config_parser.get_model_config()        
        wait_second = self.config_parser.get_wait_second()
        _gridNum = self.config_parser.get_grid_num()        
        _overlap_ratio = self.config_parser.get_overlap_ratio()
        if(model_config is not None):
            yolo_model = yoloModel(model_config)
            self.dict_yolo_model = yolo_model.getYoloModel()
            self.yolo_classes = yolo_model.getClasses()
            self.yolo_thresh = yolo_model.getThresh()

            if(len(model_config) == len(self.dict_yolo_model)):
                print(f'success loaded {len(model_config)} model')
                if ONLY_PROCESS == True:
                    self.sendMessage(f'{len(model_config)} model(s) are loaded..')

                while True:
                    if ONLY_PROCESS == True and self.mysocket is not None:
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

                    #Process Folder------------------------------------------
                    folderlist = os.listdir(self.config_parser.get_input_path())                   
                    for folder in folderlist:
                       self.procssing_folder(folder)     
                    #---------------------------------------------------------         
                                    
                    
                    if ONLY_PROCESS == True and self.mysocket is not None:
                        self.sendMessage('Waiting...')
                        for i in range(0, wait_second):
                            self.mysocket.settimeout(1.0)
                            try:
                                packet, address = self.mysocket.recvfrom(1024)
                                if (pickle.loads(packet) == 'quit'):
                                    self.mysocket.sendto(pickle.dumps('quit'), (self.HOST, self.PORT))
                                    packet, address = self.mysocket.recvfrom(1024)
                                    if (pickle.loads(packet) == 'quit'):
                                        self.sendMessage('Stop OCR Process')                                        
                                        os.exit(1)
                            except socket.timeout:
                                print('timeout')

                            self.mysocket.settimeout(None)
                            time.sleep(1.0)
                    else:
                        print(' ')                        
                        print('Waiting...') 
                    time.sleep(wait_second)
                    

if __name__ == '__main__':
    ocr_process = ocrProcess()
    ocr_process.main()

