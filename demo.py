import argparse
from pathlib import Path
import cv2
import tensorflow as tf

from models.matching import Matching
from utils.utils import load_img, get_img_paths, check_args, make_matching_plot_fast, AverageTimer, pre_process
from models.zippypoint import load_ZippyPoint
from models.postprocessing import PostProcessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ZippyPoint demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Image paths
    parser.add_argument(
        '--ref_img', type=str, default='./assets/reference/1341847980.722988.png',
        help='Path to reference image')
    parser.add_argument(
        '--query_imgs', type=str, default='./assets/query',
        help='Path to query image folder.')
    # Image processing
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    # Post processing
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.0001,
        help='Keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_window', type=int, default=3,
        help='Non Maximum Suppression (NMS) window'
        ' (Must be positive)')
    # Matching config
    parser.add_argument(
        '--ratio_threshold', type=float, default=0.95,
        help='Matching ratio threshold')
    # Display/Output configuration
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')

    args = parser.parse_args()
    args = check_args(args)

    device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    config_superpoint = {
            'nms_radius': args.nms_window,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        }
    # Define models and postprocessing
    pretrained_path = Path(__file__).parent / 'models/weights'
    ZippyPoint = load_ZippyPoint(pretrained_path, input_shape = args.resize)
    post_processing = PostProcessing(nms_window=args.nms_window,
                                     max_keypoints=args.max_keypoints,
                                     keypoint_threshold=args.keypoint_threshold)

    config_matching = {
            'do_mutual_check': True,
            'ratio_threshold': args.ratio_threshold,
        }
    matching = Matching(config_matching)
    keys = ['keypoints', 'scores', 'descriptors']

    frame = load_img(args.ref_img, args.resize)
    # Padded frame tensor.
    frame_tensor, img_pad = pre_process(frame)
    scores, keypoints, descriptors = ZippyPoint(frame_tensor, False)
    scores, keypoints, descriptors = post_processing(scores, keypoints, descriptors)
    # Correct keypoint location given required padding
    keypoints -= tf.constant([img_pad[2][0], img_pad[1][0]], dtype=tf.float32)

    last_data = {'image0': frame_tensor,
                 'scores0': tf.stack(scores),
                 'keypoints0': tf.stack(keypoints),
                 'descriptors0': tf.stack(descriptors)}

    last_frame = frame
    ref_image_name = Path(args.ref_img).stem

    if args.output_dir is not None:
        print('==> Will write outputs to {}'.format(args.output_dir))
        Path(args.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not args.no_display:
        cv2.namedWindow('ZippyPoint matching', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ZippyPoint matching', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    timer = AverageTimer(newline=True)

    for frame_path in get_img_paths(args.query_imgs):
        query_image_name = Path(frame_path).stem
        frame = load_img(frame_path, args.resize)


        timer.update('data')

        frame_tensor, img_pad = pre_process(frame)
        scores, keypoints, descriptors = ZippyPoint(frame_tensor, False)
        timer.update('extraction')
        scores, keypoints, descriptors = post_processing(scores, keypoints, descriptors)
        timer.update('post-process')
        # Correct keypoint location given required padding
        keypoints -= tf.constant([img_pad[2][0], img_pad[1][0]], dtype=tf.float32)

        new_data = {'image1': frame_tensor,
                     'scores1': tf.stack(scores),
                     'keypoints1': tf.stack(keypoints),
                     'descriptors1': tf.stack(descriptors)}

        _pred = {**last_data, **new_data}

        pred = matching(_pred)
        timer.update('matching')

        kpts0 = _pred['keypoints0'][0].cpu().numpy()
        kpts1 = _pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        text = [
            'ZippyPoint',
            'Keypoints: {} - {}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        small_text = [
            'Image Pair: {} - {}'.format(ref_image_name, query_image_name),
            'Keypoint Threshold: {:.4f}'.format(config_superpoint['keypoint_threshold']),
            'Match Threshold: {:.2f}'.format(config_matching['ratio_threshold']),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, text,
            path=None, show_keypoints=args.show_keypoints, small_text=small_text)

        if not args.no_display:
            cv2.imshow('ZippyPoint matching', out)
            key = chr(cv2.waitKey(1) & 0xFF)

        if args.output_dir is not None:
            stem = 'matches_{}_{}'.format(ref_image_name, query_image_name)
            out_file = str(Path(args.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

        timer.update('viz')
        timer.print()
    cv2.destroyAllWindows()
