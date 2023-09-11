import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import cv2

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

class VTONDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        '''
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        '''
        self.warproot = data_root['warproot']
        self.dataroot = data_root['dataroot']
        self.im_names, self.c_names = [], []
        self.datalist = 'train_pairs'
        with open(os.path.join(self.dataroot, self.datalist+'.txt'), 'r') as f:
            '''
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                self.im_names.append(im_name)
                self.c_names.append(c_name)
            '''
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                self.im_names.append(im_name)
                self.c_names.append(c_name)
                #self.c_names.append(im_name.replace('0.jpg', '1.jpg'))

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])   

        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        '''
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret
        '''
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # person image
        im = Image.open(os.path.join(self.dataroot, 'image', im_name))
        im = self.transform(im) # [-1, 1]

        # tryon image
        tryon = Image.open(os.path.join(self.warproot, 'tryon', im_name))
        tryon = self.transform(tryon) # [-1, 1]
        
        cm = Image.open(os.path.join(self.warproot, 'warp-mask', c_name.replace('.jpg','_mask.jpg')))
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0 or 1]
        cm.unsqueeze_(0)

        path = os.path.join(self.warproot, 'warp-mask', c_name.replace('.jpg','_mask.jpg'))
        image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        kernel = np.ones((5, 5), np.uint8)
        cm1 = cv2.erode(image, kernel ,iterations=3)
        cm_array1 = np.array(cm1)
        cm_array1 = (cm_array1 >= 128).astype(np.float32)
        cm1 = torch.from_numpy(cm_array1) # [0 or 1]
        cm1.unsqueeze_(0)

        cm2 = cm*(1.-cm1)

        parse_name = im_name
        im_parse = Image.open(os.path.join(self.warproot, 'segmt', parse_name.replace('front.png','segmt.png')))
        parse_array = np.array(im_parse)
        parse_arm = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32) + (parse_array == 5).astype(np.float32)
        + (parse_array == 6).astype(np.float32) + (parse_array == 7).astype(np.float32) + (parse_array == 10).astype(np.float32)
        + (parse_array == 11).astype(np.float32)
        mask_body = torch.from_numpy((parse_arm > 0).astype(np.float32)).unsqueeze(0)

        #cm2 = cm2*mask_body
        tryon1 = tryon*(1. - cm2) + cm2*torch.randn_like(tryon)

        im_hand = Image.open(os.path.join(self.dataroot, 'palm-rgb', im_name.replace('_whole_front.png','_palm.png')))
        im_hand = self.transform(im_hand)
        body_m = im_hand*mask_body + im*(1 - mask_body)

        result = {
            'c_name':               c_name, 
            'im_name':             im_name,   
            'person':                   im,
            'tryon':                tryon1,
            'mask':                    cm2,
            'mask_body':         mask_body,
            'mask_before':              cm,
            'tryon_':                tryon,
            'body_forview':         body_m,
            }

        return result 

    def __len__(self):
        return len(self.im_names)

