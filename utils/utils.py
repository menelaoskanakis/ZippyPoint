from pathlib import Path
import time
from collections import OrderedDict
import numpy as np
import cv2
from os.path import isdir
import tensorflow as tf

GLOBS = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def load_img(img_path, resize):
    """ Read image and resize to img_size.
    Inputs
        img_path: Path to input image.
    Returns
        rgmim: numpy array sized H x W x 3.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise Exception('Error reading image {}'.format(img_path))
    w, h = img.shape[1], img.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    img = cv2.resize(
        img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return img

def get_img_paths(path):
    paths = []
    for glob in GLOBS:
        paths += list(Path(path).resolve().glob(glob))
    paths = sorted([str(i) for i in paths])
    if len(paths) == 0:
        raise ValueError('Could not find any image in: {}.'.format(path))
    print('Found {} images in folder {}.'.format(len(paths), path))
    return paths

def check_args(args):
    # Check validity of ref img
    if args.ref_img is '' or args.ref_img.endswith(tuple(GLOBS)):
        raise Exception("Please specify '--ref_img' image path with "
                        "an apparopriate image and file format {}".format(args.ref_img))
    else:
        print('You specified {} as the reference image.'.format(args.ref_img))

    # Check validity of query folder
    if not isdir(Path(args.query_imgs).resolve()):
        raise Exception("Specify the folder path '--query_imgs' that includes the query "
                        "images".format(args.query_imgs))
    else:
        print('You specified {} as the folder with the query images.'.format(args.query_imgs))

    if len(args.resize) == 2 and args.resize[1] == -1:
        args.resize = args.resize[0:1]
    if len(args.resize) == 2:
        print('Will resize image to {}x{} (WxH)'.format(
            args.resize[0], args.resize[1]))
    elif len(args.resize) == 1 and args.resize[0] > 0:
        print('Will resize max dimension of image to {}'.format(args.resize[0]))
    elif len(args.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for image --resize')

    return args

def pre_process(img):
    img = frame2tensor(img)
    img = normalize(img)

    # Pad image to ensure correct dimensions for Inverted Pixel Shuffle
    H, W = img.shape[1:3]
    delta_0 = int((tf.math.ceil(H / 8) * 8)) - H
    delta_1 = int((tf.math.ceil(W / 8) * 8)) - W
    img_pad = [[0, 0],
               [delta_0 // 2, delta_0 - delta_0 // 2],
               [delta_1 // 2, delta_1 - delta_1 // 2],
               [0, 0]]
    img = tf.pad(img, img_pad)
    return img, img_pad

def normalize(img):
    img -= 0.5
    img *= 0.225
    return img

def frame2tensor(frame):
    return tf.convert_to_tensor(frame/255.)[None]

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    c = [0, 0, 120]
    H0, W0, C0 = image0.shape
    H1, W1, C1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, C0), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()
