import PySimpleGUI as sg

from sort import *
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
from tool.utils import *

"""hyper parameters"""
use_cuda = True


def track_cv2_video(cfgfile, weightfile, filename, inpath='data/', outpath='data/output/'):
    # create instance of YOLO and SORT
    m = Darknet(cfgfile)
    tracker = Sort(max_age=5, iou_threshold=0.2)

    m.print_network()
    print('Loading weights from %s...' % weightfile, end=' ')
    m.load_weights(weightfile)
    print('Done!')

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(inpath + filename)
    cap.set(3, cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(4, cv2.CAP_PROP_FRAME_WIDTH)

    print("Starting the YOLOv4+SORT loop...")

    class_names = load_class_names('data/pedestrian.names')
    i = 0
    total_fps = 0

    # Define interface
    layout = [[sg.Checkbox('Draw BBoxes', default=True, key='-checkbox-')],
              [sg.Text('Счётчик FPS: 0', key='-fps-', size=(20, 1))],
              [sg.Image(filename='', key='-image-')],
              [sg.Button('Exit', size=(7, 1), pad=((600, 0), 3), font='Helvetica 14')]]
    sg.theme('SystemDefaultForReal')
    window = sg.Window("Image Viewer", layout)

    # Starting main loop
    while True:

        # Define variables and timeout for interface
        event, values = window.read(timeout=0)

        ret, img = cap.read()

        if event in (None, 'Exit', 'Cancel') or ret is False:
            cap.release()
            window.close()
            return total_fps, i

        start = time.time()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(m, sized, 0.5, 0.6, use_cuda)

        _, boxes_with_conf = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        # SORT tracker
        if len(boxes_with_conf) == 0:
            track_bbs_ids, trackers = tracker.update()
        else:
            track_bbs_ids, trackers = tracker.update(dets=boxes_with_conf)

        # Draw BBoxes
        if values['-checkbox-']:
            img = plot_tracks_cv2(img, track_bbs_ids)
        finish = time.time()
        print('Predicted YOLOv4 in %f seconds.' % (finish - start))

        # Count FPS
        fps = round(1.0 / (finish - start), 2)
        total_fps += fps

        print('Computed SORT in %f seconds.' % (finish - start))

        cv2.imwrite(filename=outpath + str(i) + '.jpg', img=img)
        i += 1

        img = cv2.resize(img, (1280, 640))

        # Update image in interface
        image_elem = window['-image-']
        fps_elem = window['-fps-']

        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        image_elem.update(data=imgbytes)
        fps_elem.update(value='Счётчик FPS: ' + str(fps))


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str,
                        default='cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='weights/yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-inpath', type=str,
                        default='data/',
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

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    total_fps, count_frames = track_cv2_video(cfgfile,
                                              weightfile,
                                              filename='jaad/video_0031.mp4',
                                              inpath=inpath,
                                              outpath=outpath)

    print('Mean FPS: ' + str(round(total_fps / count_frames, 2)))
