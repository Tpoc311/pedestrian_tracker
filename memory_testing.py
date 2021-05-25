from memory_profiler import memory_usage

from main import get_args
from sort import *
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
from tool.utils import *

"""hyper parameters"""
use_cuda = True


def mem_test(cfgfile, weightfile, filename, w, h, inpath='data/input/'):
    # create instance of YOLO and SORT
    m = Darknet(cfgfile)
    tracker = Sort(max_age=5, iou_threshold=0.2)

    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(inpath + filename)
    cap.set(3, cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(4, cv2.CAP_PROP_FRAME_WIDTH)

    class_names = load_class_names('data/pedestrian.names')

    # Starting main loop

    while True:
        ret, img = cap.read()

        if ret is False:
            cap.release()
            return

        sized = cv2.resize(img, (w, h))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)

        _, boxes_with_conf = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        # SORT tracker
        if len(boxes_with_conf) == 0:
            tracker.update()
        else:
            tracker.update(dets=boxes_with_conf)


if __name__ == '__main__':
    args = get_args()
    import os

    cfgfile = args.cfgfile
    weightfile = args.weightfile
    inpath = args.inpath
    outpath = args.outpath

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    mem_test(args.cfgfile,
             args.weightfile,
             filename='video_0031.mp4',
             w=416,
             h=320,
             inpath='data/jaad/')
    print(memory_usage())
