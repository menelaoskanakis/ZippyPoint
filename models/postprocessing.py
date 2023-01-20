import tensorflow as tf
from models.modules.modules import SimpleNMS


class PostProcessing:
    def __init__(self, nms_window:int = 3, keypoint_threshold: float = 0.0001, max_keypoints: int = -1):
        self.select_keypoints = True if (keypoint_threshold > 0. or max_keypoints > -1) else False
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms = SimpleNMS(nms_window) if nms_window > 0 else None

    def _select_keypoints(self, scores, keypoints, descriptors):
        scoresNew = []
        keypointsNew = []
        descriptorsNew = []
        for s, p, d in zip(scores, keypoints, descriptors):
            keypoints_idx = tf.squeeze(tf.where(tf.greater_equal(s, self.keypoint_threshold)), axis=-1)
            if tf.math.count_nonzero(keypoints_idx) > self.max_keypoints and self.max_keypoints != -1:
                _scores, _idx = tf.nn.top_k(s, k=self.max_keypoints, sorted=True)
                _keypoints = tf.gather(p, _idx)
                _descriptors = tf.gather(d, _idx)
            else:
                _scores = tf.gather(s, keypoints_idx)
                _keypoints = tf.gather(p, keypoints_idx)
                _descriptors = tf.gather(d, keypoints_idx)
            scoresNew.append(_scores)
            keypointsNew.append(_keypoints)
            descriptorsNew.append(_descriptors)

        return scoresNew, keypointsNew, descriptorsNew

    def __call__(self, scores, keypoints, descriptors):
        B, H, W, C = descriptors.shape
        if self.nms:
            scores = self.nms(scores)

        scores = tf.reshape(scores, [B, H * W])
        scores = tf.unstack(scores, axis=0)

        keypoints = tf.reshape(keypoints, [B, H * W, 2])
        keypoints = tf.unstack(keypoints, axis=0)

        descriptors = tf.reshape(descriptors, [B, H * W, C])
        descriptors = tf.unstack(descriptors, axis=0)

        if self.select_keypoints:
            scores, keypoints, descriptors = self._select_keypoints(scores, keypoints, descriptors)

        return scores, keypoints, descriptors
