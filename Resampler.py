import math
import numpy as np

SINGLESTROKE_FEASIBLE_REGION_TEST = (2.93, 0.05)
SINGLESTROKE_CENTER_CORRECTION_TEST = [0.0508673, -0.07289113]
MULTISTROKE_FEASIBLE_REGION_TEST = (2.54, -0.24)
MULTISTROKE_CENTER_CORRECTION_TEST = [0.00705476, 0.07499881]

SINGLESTROKE_FEASIBLE_REGION_VAL = (3.27, -0.42)
SINGLESTROKE_CENTER_CORRECTION_VAL = [0.03692385, -0.07428717]
MULTISTROKE_FEASIBLE_REGION_VAL = 3.01, -0.67
MULTISTROKE_CENTER_CORRECTION_VAL = [0.00496911, 0.06230144]


class Resampler:

    def __init__(self, min_dist=0.05, shift_factor=0.5, scale_factor=1.0, robust_normalization=False):
        self.min_dist = min_dist
        self.shift_factor = shift_factor
        self.scale_factor = scale_factor
        self.robust_normalization = robust_normalization

    def normalize_dataset(self, data, stroke_dataset, include_fingerup=False, load_mode=''):

        # Setting device parameters for a fair normalization
        if load_mode is 'test':
            if stroke_dataset == '1$':
                max_value, min_value = SINGLESTROKE_FEASIBLE_REGION_TEST
                robust_shift_factor = SINGLESTROKE_CENTER_CORRECTION_TEST
            if stroke_dataset == 'N$':
                max_value, min_value = MULTISTROKE_FEASIBLE_REGION_TEST
                robust_shift_factor = MULTISTROKE_CENTER_CORRECTION_TEST
        if load_mode is 'validation':
            if stroke_dataset == '1$':
                max_value, min_value = SINGLESTROKE_FEASIBLE_REGION_VAL
                robust_shift_factor = SINGLESTROKE_CENTER_CORRECTION_VAL
            if stroke_dataset == 'N$':
                max_value, min_value = MULTISTROKE_FEASIBLE_REGION_VAL
                robust_shift_factor = MULTISTROKE_CENTER_CORRECTION_VAL

        for i in range(len(data)):

            # Saving stroke for multistroke
            stroke_id = []
            if stroke_dataset == 'N$':
                point_dim = 3
                raw = np.array(data[i])[:, 1:-1]
                stroke_id = np.array(data[i])[:, 0]
            else:
                raw = np.array(data[i])[:, :-1]
                point_dim = 2

            # Saving the finger up
            finger_up = []
            if include_fingerup:
                point_dim += 1
                finger_up = np.array(data[i])[:, -1]

            if load_mode is 'train':

                # Mapping the gesture bounding box in 0-1
                max = np.amax(raw, axis=0)
                min = np.amin(raw, axis=0)
                centroid = min + (max - min) / 2
                raw = raw - centroid
                raw = raw / (raw.max() * 2)

                # Generating random normal scale
                scale_mu, scale_sigma = 0.2, 0.045  # Validation tuned parameters
                s = np.random.normal(scale_mu, scale_sigma, size=1)
                raw = raw * s

                # Generating random normal shift
                shift_mu, shift_sigma = 0.5, 0.04   # Validation tuned parameters
                s = np.random.normal(shift_mu, shift_sigma, size=(1, 2))
                raw = raw + s

                # Bounding box correction
                shifted_max = np.max(raw, axis=0)
                shifted_min = np.min(raw, axis=0)
                mask_max = (shifted_max > 1)
                mask_min = (shifted_min < 0)
                if mask_max.any() or mask_min.any():

                    shifted_max[0] = np.max([shifted_max[0], 1])
                    shifted_max[1] = np.max([shifted_max[1], 1])
                    shifted_min[0] = np.min([shifted_min[0], 0])
                    shifted_min[1] = np.min([shifted_min[1], 0])

                    raw = raw + (1-shifted_max)
                    raw = raw - shifted_min

            else:
                centroid = min_value + (max_value - min_value) / 2
                raw = raw - centroid
                raw = raw / (max_value * 2)
                raw = raw + self.shift_factor
                raw = raw + np.array(robust_shift_factor)

            raw_complete = np.zeros(shape=(raw.shape[0], point_dim))

            if stroke_dataset == 'N$':
                for j in range(len(raw)):
                    if include_fingerup:
                        raw_complete[j] = [stroke_id[j], raw[j][0], raw[j][1], finger_up[j]]
                    else:
                        raw_complete[j] = [stroke_id[j], raw[j][0], raw[j][1]]

                data[i] = raw_complete

            else:
                for j in range(len(raw)):
                    if include_fingerup:
                        raw_complete[j] = [raw[j][0], raw[j][1], finger_up[j]]
                    else:
                        raw_complete[j] = [raw[j][0], raw[j][1]]

                data[i] = raw_complete

        return data

    def resample_onedollar_dataset(self, dataset, include_fingerup=False):

        for i in range(len(dataset)):
            raw = np.array((dataset[i])[:, :-1])
            if include_fingerup:
                fingerup = np.array(dataset[i])[:, -1]
                raw = (self.resample_gesture_2(raw, stroke='1$', fingerup=fingerup))
            else:
                raw = (self.resample_gesture_2(raw, stroke='1$'))

            raw = np.array(raw)
            dataset[i] = raw

        return dataset

    def resample_ndollar_dataset(self, dataset, include_fingerup=False):

        for i in range(len(dataset)):
            raw = np.array(dataset[i])[:, 1:-1]
            stroke_id = np.array(dataset[i])[:, 0]
            if include_fingerup:
                fingerup = np.array(dataset[i])[:, -1]
                raw = (self.resample_gesture_2(raw, stroke='N$', stroke_ids=stroke_id, fingerup=fingerup))
            else:
                raw = (self.resample_gesture_2(raw, stroke='N$', stroke_ids=stroke_id))

            raw = np.array(raw)
            dataset[i] = raw

        return dataset

    def resample_gesture_2(self, raw, stroke='1$', stroke_ids=[], fingerup=[]):

        resampled = []
        self.min_dist = 0.01

        xs, ys, raw_xs, raw_ys = [], [], [], []

        pivot = raw[0]
        D, i = 0, 0
        while i < len(raw)-1:

            raw_xs.append(raw[i][0])
            raw_ys.append(raw[i][1])
            d = self.__get_dist(pivot, raw[i + 1])

            if D + d < self.min_dist:
                pivot = raw[i+1]
                D += d
            else:
                rest = D+d - self.min_dist
                travel_dist = d - rest
                delta = raw[i+1]-pivot
                alpha = math.atan2(delta[1], delta[0])
                new_x = travel_dist * math.cos(alpha) + pivot[0]
                new_y = travel_dist * math.sin(alpha) + pivot[1]

                if not (stroke == 'N$' and stroke_ids[i+1] == stroke_ids[i] + 1):
                    pivot = [new_x, new_y]
                else:
                    pivot = raw[i+1]

                xs.append(new_x)
                ys.append(new_y)
                if stroke == 'N$':
                    if fingerup is not []:
                        resampled.append([stroke_ids[i], pivot[0], pivot[1], fingerup[i]])
                    else:
                        resampled.append([stroke_ids[i], pivot[0], pivot[1]])
                else:
                    if fingerup is not []:
                        resampled.append([pivot[0], pivot[1], fingerup[i]])
                    else:
                        resampled.append(pivot)

                raw[i] = pivot
                i -= 1
                D = 0

            i += 1

        if stroke == 'N$':
            if fingerup is not []:
                resampled.append([stroke_ids[-1], raw[-1][0], raw[-1][1], fingerup[-1]])
            else:
                resampled.append([stroke_ids[-1], raw[-1][0], raw[-1][1]])
        else:
            if fingerup is not []:
                resampled.append([raw[-1][0], raw[-1][1], fingerup[-1]])
            else:
                resampled.append([raw[-1][0], raw[-1][1]])

        return resampled

    def __get_dist(self, p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return math.sqrt(dy * dy + dx * dx)

    def _path_length(self, gesture):
        d = 0.0
        for i in range(len(gesture)-1):
            d += self._distance(gesture[i], gesture[i+1])
        return d

    def _distance(self, p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return math.sqrt(dx*dx + dy*dy)
