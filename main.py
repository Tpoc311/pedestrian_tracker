# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    : Tpoc311
    @File      : main.py (renamed)
    @Time      :
    @Detail    :
'''
import argparse

from sort import *
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
from tool.utils import *

"""hyper parameters"""
use_cuda = True


def track_cv2_video(cfgfile, weightfile, filename, inpath='data/input/', outpath='data/output/'):
    import cv2
    m = Darknet(cfgfile)

    # create instance of SORT
    mot_tracker = Sort(max_age=5, iou_threshold=0.8)

    m.print_network()
    print('Loading weights from %s...' % weightfile, end=' ')
    m.load_weights(weightfile)
    print('Done!')

    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(inpath + filename)
    cap.set(3, cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(4, cv2.CAP_PROP_FRAME_WIDTH)
    print("Starting the YOLOv4+SORT loop...")

    class_names = load_class_names('data/pedestrian.names')

    history = {}
    while True:
        ret, img = cap.read()

        if ret is False:
            print("End of video")
            cap.release()
            return

        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)
        finish = time.time()
        print('Predicted YOLOv4 in %f seconds.' % (finish - start))

        result_img, boxes_with_conf = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        # SORT tracker
        start = time.time()
        track_bbs_ids, trackers = mot_tracker.update(boxes_with_conf)
        img_with_tracks = plot_tracks_cv2(img, track_bbs_ids)

        # TODO Drawing trajectories
        img_with_tracks = print_trajectories(img_with_tracks, trackers)
        # TODO END

        finish = time.time()
        print('Computed SORT in %f seconds.' % (finish - start))

        cv2.imshow('YOLOv4+SORT demo ', img_with_tracks)
        cv2.waitKey(1)

    cap.release()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str,
                        default='cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='weights/yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-inpath', type=str,
                        default='data/input/',
                        help='folder containing input images', dest='inpath')
    parser.add_argument('-outpath', type=str,
                        default='data/output/',
                        help='folder which will contain images with detections', dest='outpath')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    import os

    cfgfile = args.cfgfile
    weightfile = args.weightfile
    inpath = args.inpath
    outpath = args.outpath

    if not os.path.exists(inpath):
        os.mkdir(inpath)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    track_cv2_video(cfgfile,
                    weightfile,
                    filename='mot.webm',
                    inpath=inpath,
                    outpath=outpath)
# mot.webm
# /JAAD_clips/video_0002.mp4
