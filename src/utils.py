import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon
import torch.nn as nn

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    if mask_y >= mask_y + mask_height or mask_x >= mask_x + mask_width :
        print(mask_y, mask_y + mask_height, mask_x, mask_x + mask_width)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def mask_generation_with_BB(img_shape, BB, offset=0):
    """
    Generate a word-level for a image with given 2D word bounding box
    """
    mask = np.zeros(img_shape[0:2], dtype=np.float32)
    nb_instance = BB.shape[-1]
    for i in range(nb_instance):
        bb = BB[:, :, i]
        bb = np.transpose(bb, [1, 0])
        # Enlarge the bounding box
        bb = bb + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * offset
        # bb1 = np.clip(bb[:, 1], 0, img_shape[0])
        # bb0 = np.clip(bb[:, 0], 0, img_shape[1])
        bb1 = bb[:, 1]
        bb0 = bb[:, 0]
        rr, cc = polygon(bb1, bb0, img_shape)
        mask[rr, cc] = 1.

    return mask

def mask_generation_with_BB_random(img_shape, BB, offset=0):
    nb_instance = BB.shape[-1]
    index_not_selected = np.random.permutation(nb_instance)[:np.random.choice(nb_instance)]
    index_all = np.array(range(nb_instance))
    index_selected = np.setxor1d(index_not_selected, index_all)
    BB_not_selected = BB[..., index_not_selected]
    BB2_selected = BB[..., index_selected]
    mask_not_selected = mask_generation_with_BB(img_shape, BB_not_selected, 0)
    mask_selected = mask_generation_with_BB(img_shape, BB2_selected, offset)
    return np.multiply(mask_selected, 1 - mask_not_selected)

def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns +
                            gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()

def imread(path):
    im = Image.open(path)
    return np.array(im)

def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)

def output_align(input, output):
    """
    In testing, sometimes output is several pixels less than irregular-size input,
    here is to fill them
    """
    if output.size() != input.size():
        diff_width = input.size(-1) - output.size(-1)
        diff_height = input.size(-2) - output.size(-2)
        m = nn.ReplicationPad2d((0, diff_width, 0, diff_height))
        output = m(output)

    return output

def random_size(length, ratio=0.6):
    """ used for random crop."""
    ratio = random.uniform(ratio, 1)
    min_value = random.randint(0, np.floor(length * (1- ratio)))
    max_value = min_value + np.floor(length * ratio).astype(np.int)
    return min_value, max_value

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, interval=0.05,
                 stateful_metrics=None):
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._cur_values = {}
        self._avg_values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._info = ''

    # def update(self, current, values):
    #     """Updates the progress bar.
    #
    #     Arguments:
    #         current: Index of current step.
    #         values: List of tuples:
    #             `(name, value_for_last_step)`.
    #             If `name` is in `stateful_metrics`,
    #             `value_for_last_step` will be displayed as-is.
    #             Else, an average of the metric over time will be displayed.
    #     """
    #     values = values or []
    #     for k, v in values:
    #         if k not in self._values_order:
    #             self._values_order.append(k)
    #         if k not in self.stateful_metrics:
    #             if k not in self._avg_values:
    #                 self._values[k] = [v * (current - self._seen_so_far),
    #                                    current - self._seen_so_far]
    #             else:
    #                 self._values[k][0] += v * (current - self._seen_so_far)
    #                 self._values[k][1] += (current - self._seen_so_far)
    #             self._cur_values[k] = v
    #         else:
    #             self._values[k] = v
    #             self._cur_values[k] = v
    #     self._seen_so_far = current
    #
    #     #info = ' - %.0fs' % (now - self._start)
    #     avg_info = ''
    #     cur_info = ''
    #
    #     for k in self._values_order:
    #         avg_info += ' %s=' % k
    #         cur_info += ' %s=' % k
    #         if isinstance(self._values[k], list):
    #             avg = np.mean(self._avg_values[k][0] / max(1, self._avg_values[k][1]))
    #             self._avg_values[k] = avg
    #             # avg = self._values[k][0] / max(1, self._values[k][1])
    #             if abs(avg) > 1e-3:
    #                 avg_info += '%.4f' % avg
    #             else:
    #                 avg_info += '%.4e' % avg
    #
    #             if abs(self._cur_values[k]) > 1e-3:
    #                 cur_info += ' %s' % self._cur_values[k]
    #             else:
    #                 cur_info += ' %s' % self._cur_values[k]
    #         else:
    #             # ???
    #             avg_info += ' %s' % self._values[k]
    #             cur_info += ' %s' % self._cur_values[k]
    #
    #     avg_info += ' ('
    #     cur_info += ' ('
    #     now = time.time()
    #     if current:
    #         time_per_unit = ((now - self._start) / current) * len(values)
    #     else:
    #         time_per_unit = 0
    #
    #     if time_per_unit >= 1:
    #         avg_info += '%.0fs/step' % time_per_unit
    #         cur_info += '%.0fs/step' % time_per_unit
    #     elif time_per_unit >= 1e-3:
    #         avg_info += '%.0fms/step' % (time_per_unit * 1e3)
    #         cur_info += '%.0fms/step' % (time_per_unit * 1e3)
    #     else:
    #         avg_info += '%.0fus/step' % (time_per_unit * 1e6)
    #         cur_info += '%.0fus/step' % (time_per_unit * 1e6)
    #
    #     avg_info += ')'
    #     avg_info += '\n'
    #     cur_info += ')'
    #     cur_info += '\n'
    #
    #     self.avg_info = avg_info
    #     self.cur_info = cur_info
    #     self._last_update = now

    def update(self, current, values):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        # info = ' - %.0fs' % (now - self._start)
        info = ''

        for k in self._values_order:
            info += ' %s=' % k
            if isinstance(self._values[k], list):
                avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                self._avg_values[k] = avg
                # avg = self._values[k][0] / max(1, self._values[k][1])
                if abs(avg) > 1e-3:
                    info += '%.4f' % avg
                else:
                    info += '%.4e' % avg
            else:
                info += '%s' % self._values[k]

        info += ' ('
        now = time.time()
        if current:
            time_per_unit = (now - self._start) / current
        else:
            time_per_unit = 0

        if time_per_unit >= 1:
            info += '%.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
            info += '%.0fms/step' % (time_per_unit * 1e3)
        else:
            info += '%.0fus/step' % (time_per_unit * 1e6)

        info += ')'
        info += '\n'

        self._info = info
        self._last_update = now

    def print_cur(self, current, values):
        """Print the current information"""
        cur_info = ''

        for k, v in values:
            cur_info += ' %s=' % k

            if k not in self.stateful_metrics:
                if abs(v) > 1e-3:
                    cur_info += '%.4f' % v
                else:
                    cur_info += '%.4e' % v
            else:
                cur_info += '%s' % v

        cur_info += ' ('
        now = time.time()
        if current:
            #print(now - self._last_update, current, len(values[0]))
            time_per_unit = (now - self._last_update) / current
        else:
            time_per_unit = 0

        if time_per_unit >= 1:
            cur_info += '%.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
            cur_info += '%.0fms/step' % (time_per_unit * 1e3)
        else:
            cur_info += '%.0fus/step' % (time_per_unit * 1e6)

        cur_info += ')'
        cur_info += '\n'

        self.cur_info = cur_info
        self._last_update = now

        sys.stdout.write(cur_info)
        sys.stdout.flush()

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def print_info(self):
        sys.stdout.write(self._info)
        sys.stdout.flush()

    def get_average_log_values(self):
        return self._avg_values