# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : main.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''
import cv2

from sort import *
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *

"""hyper parameters"""
use_cuda = True


def detect_cv2_folder(cfgfile, weightfile, in_path, out_path):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/pedestrian.names'
    class_names = load_class_names(namesfile)

    for filename in os.listdir(in_path):
        file_path = os.path.join(in_path + filename)

        img = cv2.imread(file_path)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (filename, (finish - start)))

        img, _ = plot_boxes_cv2(img, boxes[0], class_names=class_names)

        cv2.imwrite(out_path + filename, img)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/pedestrian.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    return plot_boxes_cv2(img, boxes[0], class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-inpath', type=str,
                        default='data/images',
                        help='folder containing input images', dest='inpath')
    parser.add_argument('-outpath', type=str,
                        default='data/output',
                        help='folder which will contain output images', dest='outpath')
    args = parser.parse_args()

    return args


# def get_detections():

def plot_tracks(img, boxes):
    width = img.shape[1]
    height = img.shape[0]

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        id = str(box[4])

        rgb = (255, 0, 0)
        img = cv2.putText(img, id, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)

    cv2.imshow('Yolo demo', img)

    print(11)


if __name__ == '__main__':
    args = get_args()
    import os

    cfgfile = args.cfgfile
    weightfile = args.weightfile
    inpath = args.inpath
    outpath = args.outpath

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    detect_cv2_folder(cfgfile,
                      weightfile,
                      inpath,
                      outpath)

    # TODO SORT Tracker
    # ##############################################################################################
    # create instance of SORT
    # mot_tracker = Sort()
    # import cv2
    #
    # m = Darknet(cfgfile)
    #
    # m.print_network()
    # m.load_weights(weightfile)
    # print('Loading weights from %s... Done!' % (weightfile))
    #
    # if use_cuda:
    #     m.cuda()
    #
    # if not os.path.exists(outpath):
    #     os.mkdir(outpath)
    #
    # num_classes = m.num_classes
    # if num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif num_classes == 80:
    #     namesfile = 'data/coco.names'
    # else:
    #     namesfile = 'data/pedestrian.names'
    # class_names = load_class_names(namesfile)
    #
    # for filename in os.listdir(inpath):
    #     file_path = os.path.join(inpath + filename)
    #
    #     img = cv2.imread(file_path)
    #     sized = cv2.resize(img, (m.width, m.height))
    #     sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    #
    #     for i in range(2):
    #         start = time.time()
    #         boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)
    #         finish = time.time()
    #
    #         if i == 1:
    #             print('%s: Predicted in %f seconds.' % (filename, (finish - start)))
    #     to_plot, bboxes = plot_boxes_cv2(img, boxes[0], class_names=class_names)
    #
    #     # update SORT
    #     # track_bbs_ids = mot_tracker.update(bboxes)
    #     # plot_tracks(img, track_bbs_ids)
    #
    #     cv2.imwrite(outpath + filename, to_plot)
    # ##############################################################################################

    # if args.imgfile:
    #     detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
    #     # detect_imges(args.cfgfile, args.weightfile)
    #     # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
    #     # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    # else:
    #     detect_cv2_camera(args.cfgfile, args.weightfile)