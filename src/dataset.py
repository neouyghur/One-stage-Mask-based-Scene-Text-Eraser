import os
import glob
import scipy
import torch
import random
import numpy as np
import json
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
# from scipy.misc import imread
import skimage
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask, mask_generation_with_BB, imread, random_size
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = 1000000000

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, afile, augment=True, training=True):
        """

        Args:
            gt_list: groundtruth list
        """
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_data(afile, config.WORD_BB_PERCENT_THRESHOLD)
        self._mask_pad = config.MASK_PAD
        self._mask_safe_pad = config.MASK_SAFE_PAD
        self._mask_pad_update_step = config.MASK_PAD_UPDATE_STEP

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.mask = config.MASK
        self.mask_threshold = config.MASK_THRESHOLD
        self.nms = config.NMS
        self._count = 0
        self.backup_item = None

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 7

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # item = self.load_item(index)
        try:
            # item = self.load_item(index)
            # item = self.load_item_SCUT(index)
            item = self.load_item_raindrop(index)
            self.backup_item = item
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            """Handling errors introduced by random mask generation step introduced in dataloader."""
            print('loading error: item ' + str(index))
            # item = self.__getitem__(index+1)
            if self.backup_item is not None:
                item = self.backup_item
            else:
                item = self.__getitem__(index + 1)

        return item

    def load_name(self, index):
        name = self.data[index]['dir']
        return os.path.basename(name)

    def load_item(self, index):
        self._count += 1
        size = self.input_size

        # load image
        img = imread(self.data[index]['dir'])
        if os.path.exists(self.data[index]['gt_dir']):
            img_gt = imread(self.data[index]['gt_dir'])
        else:
            img_gt = imread(self.data[index]['gt_dir'].split(".")[0] + '.png', mode='RGB')

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        if len(img_gt.shape) < 3:
            img_gt = gray2rgb(img_gt)

        # load mask
        masks_pad, masks_gt = self.load_mask(img, index)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img.astype(np.uint8), size, size)
            img_gt = self.resize(img_gt.astype(np.uint8), size, size)
            masks_pad = self.resize(masks_pad.astype(np.uint8), size, size)
            masks_gt = self.resize(masks_gt.astype(np.uint8), size, size)


        if np.mean(masks_pad) == 0 or np.mean(masks_gt) ==0:
                raise

        if img.shape != img_gt.shape:
            img_gt = self.resize(img_gt.astype(np.uint8), img.shape[0], img.shape[1])
            # print(masks_pad.shape, img_gt.shape, img.shape)

        # augment data: horizontal flip
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gt = img_gt[:, ::-1, ...]
            masks_pad = masks_pad[:, ::-1, ...]
            masks_gt = masks_gt[:, ::-1, ...]

        # !!! Has to change the type, UINT8 subtraction is different
        masks_refine_gt = np.greater(np.mean(np.abs(img.astype(np.float32) - img_gt.astype(np.float32)), axis=-1),
                                   self.mask_threshold).astype(np.uint8)

        masks_refine_gt = np.multiply(masks_refine_gt, masks_gt)
        img_gt = np.multiply(np.expand_dims((1-masks_gt), -1), img) + np.multiply(np.expand_dims(masks_gt, -1), img_gt)

        return self.to_tensor(img), self.to_tensor(img_gt), \
               self.to_tensor(masks_pad.astype(np.float64)), self.to_tensor(masks_gt.astype(np.float64)), \
               self.to_tensor(masks_refine_gt.astype(np.float64))


    # def load_item_raindrop(self, index):
    #     self._count += 1
    #     size = self.input_size
    #
    #     # load image
    #     img = imread(self.data[index]['dir'])
    #     if os.path.exists(self.data[index]['gt_dir']):
    #         img_gt = imread(self.data[index]['gt_dir'])
    #     else:
    #         img_gt = imread(self.data[index]['gt_dir'].split(".")[0] + '.png', mode='RGB')
    #
    #     # random crop
    #     left, rigth = random_size(img.shape[0], 0.8)
    #     top, bottom = random_size(img.shape[1], 0.8)
    #     img = img[left:rigth, top: bottom, :]
    #     img_gt = img_gt[left:rigth, top: bottom, :]
    #     #
    #
    #     # gray to rgb
    #     if len(img.shape) < 3:
    #         img = gray2rgb(img)
    #
    #     if len(img_gt.shape) < 3:
    #         img_gt = gray2rgb(img_gt)
    #
    #     # resize/crop if needed
    #     if size != 0:
    #         img = self.resize(img.astype(np.uint8), size, size)
    #         img_gt = self.resize(img_gt.astype(np.uint8), size, size)
    #
    #     # load mask
    #     masks_pad, masks_gt = self.load_mask(img, index)
    #     masks_pad = masks_pad / 255
    #     masks_gt = masks_gt / 255
    #
    #     if np.mean(masks_pad) == 0 or np.mean(masks_gt) ==0:
    #         raise
    #
    #     if img.shape != img_gt.shape:
    #         img_gt = self.resize(img_gt.astype(np.uint8), img.shape[0], img.shape[1])
    #         # print(masks_pad.shape, img_gt.shape, img.shape)
    #
    #     # augment data: horizontal flip
    #     if self.augment and np.random.binomial(1, 0.5) > 0:
    #         img = img[:, ::-1, ...]
    #         img_gt = img_gt[:, ::-1, ...]
    #         masks_pad = masks_pad[:, ::-1, ...]
    #         masks_gt = masks_gt[:, ::-1, ...]
    #
    #     # augment data: vertical flip
    #     if self.augment and np.random.binomial(1, 0.5) > 0:
    #         img = img[::-1, ...]
    #         img_gt = img_gt[::-1, ...]
    #         masks_pad = masks_pad[::-1, ...]
    #         masks_gt = masks_gt[::-1, ...]
    #
    #     # # !!! Has to change the type, UINT8 subtraction is different
    #     # diff = np.mean(np.abs(img_gt.astype(np.float32) - img.astype(np.float32)), axis=-1)
    #     # mask_threshold = np.mean(diff)
    #     # masks_refine_gt = np.greater(diff, mask_threshold).astype(np.uint8)
    #     #
    #     # # Remove small white regions
    #     # open_img = ndimage.binary_opening(masks_refine_gt)
    #     # # Remove small black hole
    #     # masks_refine_gt = ndimage.binary_closing(open_img)
    #
    #     # masks_refine_gt = np.multiply(masks_refine_gt, masks_gt)
    #     # img_gt = np.multiply(np.expand_dims((1-masks_gt), -1), img) + np.multiply(np.expand_dims(masks_gt, -1), img_gt)
    #     # img_gt = img_gt.astype(np.uint8)
    #
    #     masks_refine_gt = masks_gt
    #
    #     return self.to_tensor(img), self.to_tensor(img_gt), \
    #            self.to_tensor(masks_pad.astype(np.float64)), self.to_tensor(masks_gt.astype(np.float64)), \
    #            self.to_tensor(masks_refine_gt.astype(np.float64))

    def load_item_raindrop(self, index):
        self._count += 1
        size = self.input_size

        # load image
        img = imread(self.data[index]['dir'])
        if os.path.exists(self.data[index]['gt_dir']):
            img_gt = imread(self.data[index]['gt_dir'])
        else:
            img_gt = imread(self.data[index]['gt_dir'].split(".")[0] + '.png', mode='RGB')

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        if len(img_gt.shape) < 3:
            img_gt = gray2rgb(img_gt)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img.astype(np.uint8), size, size)
            img_gt = self.resize(img_gt.astype(np.uint8), size, size)

        # load mask
        masks_pad, masks_gt = self.load_mask(img, index)
        masks_pad = masks_pad / 255
        masks_gt = masks_gt / 255

        if np.mean(masks_pad) == 0 or np.mean(masks_gt) ==0:
            raise

        if img.shape != img_gt.shape:
            img_gt = self.resize(img_gt.astype(np.uint8), img.shape[0], img.shape[1])
            # print(masks_pad.shape, img_gt.shape, img.shape)

        masks_refine_gt = masks_gt

        return self.to_tensor(img), self.to_tensor(img_gt), \
               self.to_tensor(masks_pad.astype(np.float64)), self.to_tensor(masks_gt.astype(np.float64)), \
               self.to_tensor(masks_refine_gt.astype(np.float64))

    def load_item_SCUT(self, index):
        self._count += 1
        size = self.input_size

        # load image
        img = imread(self.data[index]['dir'])
        if os.path.exists(self.data[index]['gt_dir']):
            img_gt = imread(self.data[index]['gt_dir'])
        else:
            img_gt = imread(self.data[index]['gt_dir'].split(".")[0] + '.png', mode='RGB')

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        if len(img_gt.shape) < 3:
            img_gt = gray2rgb(img_gt)

        # load mask
        imgh, imgw = img.shape[0:2]
        masks_pad = np.ones([imgh, imgw])

        # resize/crop if needed
        if size != 0:
            img = self.resize(img.astype(np.uint8), size, size)
            img_gt = self.resize(img_gt.astype(np.uint8), size, size)
            masks_pad = self.resize(masks_pad.astype(np.uint8), size, size)

        if img.shape != img_gt.shape:
            img_gt = self.resize(img_gt.astype(np.uint8), img.shape[0], img.shape[1])
            # print(masks_pad.shape, img_gt.shape, img.shape)

        # augment data: horizontal flip
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gt = img_gt[:, ::-1, ...]
            masks_pad = masks_pad[:, ::-1, ...]

        # !!! Has to change the type, UINT8 subtraction is different
        masks_refine_gt = np.greater(np.mean(np.abs(img.astype(np.float32) - img_gt.astype(np.float32)), axis=-1),
                                     self.mask_threshold).astype(np.uint8)

        masks_gt = masks_refine_gt
        return self.to_tensor(img), self.to_tensor(img_gt), \
               self.to_tensor(masks_pad.astype(np.float64)), self.to_tensor(masks_gt.astype(np.float64)), \
               self.to_tensor(masks_refine_gt.astype(np.float64))

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            mask = create_mask(imgw, imgh, imgw // 2, imgh // 2)
            return mask

        if mask_type == 8:
            # print(imgw, imgh)
            # x = random.randint(imgw//4, imgw)
            # y = random.randint(imgh//4, imgh)
            # mask = create_mask(imgw, imgh, x, y)
            # if np.random.binomial(1, 0.1) > 0:
            #     mask = np.ones_like(mask)

            mask = np.ones([imgw, imgh])
            mask = (mask * 255).astype(np.uint8)
            return mask, mask

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        if mask_type == 7:
            bbox =  np.array(self.data[index]['word_bb'])
            max_pad = np.max([imgh, imgw])
            if self._mask_pad == -1:
                # coefficient = 1
                # pad = coefficient*self._count//self._mask_pad_update_step
                # if pad > np.max(self.input_size+coefficient):
                #     pad = np.random.randint(0, np.max(self.input_size), 1)[0]
                # elif pad == 0:
                #     pad = 0
                # else:
                #     pad = np.random.randint(0, pad)

                if np.random.binomial(1, 0.1) > 0:
                    pad = max_pad
                else:
                    pad = np.random.randint(self._mask_safe_pad, np.ceil(max_pad/2))

            elif self._mask_pad == -2:
                # pad = np.random.randint(2, self._mask_pad, 1)[0]
                if self.data[index]['word_percent'] < 5:
                    pad = 20
                elif self.data[index]['word_percent'] < 10:
                    pad = 15
                elif self.data[index]['word_percent'] < 15:
                    pad = 10
                else:
                    pad = 5
            else:
                pad = self._mask_pad


            if not self.training:
                return mask_generation_with_BB([imgh, imgw], bbox, pad), \
                        mask_generation_with_BB([imgh, imgw], bbox, self._mask_safe_pad)

            # return np.ones([imgh, imgw]), mask_generation_with_BB([imgh, imgw], bbox, self._mask_safe_pad)

            nb_instance = bbox.shape[-1]
            # index_selected = np.random.permutation(nb_instance)[:np.random.choice(nb_instance-1)+1]
            index_selected = np.random.permutation(nb_instance)[:nb_instance - nb_instance//5]
            index_all = np.array(range(nb_instance))
            index_not_selected = np.setxor1d(index_selected, index_all)
            #print(len(index_selected), len(index_not_selected))

            BB_not_selected = bbox[..., index_not_selected]
            BB2_selected = bbox[..., index_selected]
            mask_not_selected = mask_generation_with_BB([imgh, imgw], BB_not_selected, self._mask_safe_pad)
            mask_selected = mask_generation_with_BB([imgh, imgw], BB2_selected, self._mask_safe_pad)
            mask_safe_bbox = np.multiply(mask_selected, 1 - mask_not_selected)

            if pad >= max_pad or np.sum(mask_safe_bbox)==0:
                return np.ones([imgh, imgw]), mask_generation_with_BB([imgh, imgw], bbox, self._mask_safe_pad)
            else:
                mask_selected = mask_generation_with_BB([imgh, imgw], BB2_selected, pad)
                masks_pad = np.multiply(mask_selected, 1 - mask_not_selected)
                return masks_pad, mask_safe_bbox

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = scipy.misc.imresize(img, [height, width])
        img = skimage.transform.resize(img, [height, width])
        img = (img * 255).astype(np.uint8)

        return img

    # def load_flist(self, flist):
    #     if isinstance(flist, list):
    #         return flist
    #
    #     # flist: image file path, image directory path, text file flist path
    #     if isinstance(flist, str):
    #         if os.path.isdir(flist):
    #             flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
    #             flist.sort()
    #             return flist
    #
    #         if os.path.isfile(flist):
    #             try:
    #                 return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
    #             except:
    #                 return [flist]
    #
    #     return []

    def load_data(self, afile, theta=0):
        with open(afile) as f:
            data = json.load(f)
            if theta>0:
                name, subfix = os.path.basename(afile).split(".")
                file_path = '%s/%s_%s.%s' % (os.path.dirname(afile), name, str(theta), subfix)
                if os.path.exists(file_path):
                    print('%s is used' % (file_path))
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                else:
                    data = self.filter_data(data, theta)
                    with open(file_path, 'w') as f:
                        json.dump(data, f)
                        print('%s is created and used.' % (file_path))

            if False:
                data = data[:10]
                print("data length", len(data))
            return data

    def filter_data(self, data, theta):
        new_data = []
        for item in data:
            if item['word_percent'] >= theta:
                new_data.append(item)

        print("%d are filtered from %d data" % (len(data) -  len(new_data), len(data)))
        return new_data

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    @staticmethod
    def get_sobel_edge(img, threshold=0):
        dx = scipy.ndimage.sobel(img, 0)  # horizontal derivative
        dy = scipy.ndimage.sobel(img, 1)  # vertical derivative
        mag = np.hypot(dx, dy)  # magnitude
        return mag