import json

from main import get_args
from sort import *
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
from tool.utils import *

"""hyper parameters"""
use_cuda = True


def fps_test(cfgfile, weightfile, filename, w, h, inpath='data/input/'):
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

    count_frames = 0
    detection_fps = 0
    tracking_fps = 0
    overall_fps = 0

    # Starting main loop
    while True:
        start_overall = time.time()
        ret, img = cap.read()

        if ret is False:
            cap.release()
            mean_fps = {'mean_detection_fps': round(detection_fps / count_frames, 2),
                        'mean_tracking_fps': round(tracking_fps / count_frames, 2),
                        'mean_overall_fps': round(overall_fps / count_frames, 2)}
            return mean_fps

        sized = cv2.resize(img, (w, h))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start_detect = time.time()
        boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)
        finish_detect = time.time()

        _, boxes_with_conf = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        # SORT tracker
        start_track = time.time()
        if len(boxes_with_conf) == 0:
            tracker.update()
        else:
            tracker.update(dets=boxes_with_conf)
        finish_track = time.time()

        # Count FPS
        detection_fps += round(1.0 / (finish_detect - start_detect), 2)
        tracking_fps += round(1.0 / (finish_track - start_track), 2)
        overall_fps += round(1.0 / (finish_track - start_overall), 2)
        count_frames += 1


def count_mean_fps(input_location, w, h):
    args = get_args()
    overall_mean_fps = {'mean_detection_fps': 0.0,
                        'mean_tracking_fps': 0.0,
                        'mean_overall_fps': 0.0}
    files = os.listdir(input_location)
    files_count = 0
    for file in files:
        print(input_location + file)
        mean_fps_for_video = fps_test(args.cfgfile,
                                      args.weightfile,
                                      filename=file,
                                      w=w,
                                      h=h,
                                      inpath=input_location)
        overall_mean_fps['mean_detection_fps'] += mean_fps_for_video['mean_detection_fps']
        overall_mean_fps['mean_tracking_fps'] += mean_fps_for_video['mean_tracking_fps']
        overall_mean_fps['mean_overall_fps'] += mean_fps_for_video['mean_overall_fps']
        files_count += 1
        # shutil.move(input_location+file, "data/save/" + file)
    overall_mean_fps['mean_detection_fps'] /= files_count
    overall_mean_fps['mean_tracking_fps'] /= files_count
    overall_mean_fps['mean_overall_fps'] /= files_count

    return overall_mean_fps


def test_all(w, h):
    with open(r"testing/testing_320x416.txt", "w") as file:
        test_custom = count_mean_fps('data/custom/', w=w, h=h)
        file.write('Test data: ' + json.dumps(test_custom) + '\n')
        test_mot = count_mean_fps('data/mot/', w=w, h=h)
        file.write('MOT data: ' + json.dumps(test_mot) + '\n')
        test_jaad = count_mean_fps('data/jaad/', w=w, h=h)
        file.write('JAAD data: ' + json.dumps(test_jaad) + '\n')


test_all(w=320, h=416)
