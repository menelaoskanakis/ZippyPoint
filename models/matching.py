from copy import copy
import tensorflow as tf


class Matching:
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def __init__(self, conf):
        self.conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)

    @staticmethod
    def _find_nn(dist, ratio_thresh, distance_thresh):
        dist_nn, ind_nn = tf.nn.top_k(-1 * dist, 2 if ratio_thresh else 1, sorted=False)
        dist_nn = -1 * dist_nn
        mask = tf.ones(ind_nn.shape[:-1], dtype=tf.dtypes.bool)
        if ratio_thresh:
            mask = mask & (dist_nn[..., 1] <= (ratio_thresh) * dist_nn[..., 0])
        if distance_thresh:
            mask = mask & (dist_nn[..., -1] <= distance_thresh)
        matches = tf.where(mask, ind_nn[..., -1], -1)
        return matches

    @staticmethod
    def _gather(x, gather_axis, indices):
        """
        Gather function that is comparable to that of torch.gather()
        """

        all_indices = tf.where(tf.fill(indices.shape, True))
        gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

        gather_indices = []
        for axis in range(len(indices.shape)):
            if axis == gather_axis:
                gather_indices.append(tf.cast(gather_locations, dtype=tf.int32))
            else:
                gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int32))

        gather_indices = tf.stack(gather_indices, axis=-1)
        gathered = tf.gather_nd(x, gather_indices)
        reshaped = tf.reshape(gathered, indices.shape)
        return reshaped

    def _mutual_check(self, m0, m1):
        inds0 = tf.range(m0.shape[-1], dtype=tf.int32)
        loop = self._gather(m1, 1, tf.where(m0 > -1, m0, 0))
        ok = (m0 > -1) & (inds0 == loop)
        m0_new = tf.where(ok, m0, -1)
        return m0_new

    def __call__(self, data):
        for key in self.required_inputs:
            assert key in data, 'Missing key {} in data'.format(key)
        if data['descriptors0'].shape[-1] == 0 or data['descriptors1'].shape[-1] == 0:
            matches0 = tf.fill(
                data['descriptors0'].shape[:2], -1)
            return {
                'matches0': matches0,
                'matching_scores0': tf.zeros_like(matches0)
            }
        ratio_threshold = self.conf['ratio_threshold']
        if data['descriptors0'].shape[-1] == 1 or data['descriptors1'].shape[-1] == 1:
            ratio_threshold = None

        d0, d1 = data['descriptors0'], data['descriptors1']
        dist = tf.einsum('bnd,bmd->bnm', 1 - d0, d1) + tf.einsum('bnd,bmd->bnm', d0, 1 - d1)

        matches0 = self._find_nn(dist, ratio_threshold, self.conf['distance_threshold'])

        if self.conf['do_mutual_check']:
            dist = tf.transpose(dist, perm=[0, 2, 1])
            matches1 = self._find_nn(dist, ratio_threshold, self.conf['distance_threshold'])
            matches0 = self._mutual_check(matches0, matches1)

        return {
            'matches0': matches0,
        }
