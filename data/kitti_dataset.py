import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from data.base_dataset import BaseDataset
from models.sne_model import SNE
from PIL import Image
import random
class kittiCalibInfo():
    """
    Read calibration files in the kitti dataset,
    we need to use the intrinsic parameter of the cam2
    """
    def __init__(self, filepath):
        """
        Args:
            filepath ([str]): calibration file path (AAA.txt)
        """
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        """
        Returns:
            [numpy.array]: intrinsic parameter of the cam2
        """
        return self.data['P2']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        P0 = np.reshape(rawdata['P0'], (3,4))
        P1 = np.reshape(rawdata['P1'], (3,4))
        P2 = np.reshape(rawdata['P2'], (3,4))
        P3 = np.reshape(rawdata['P3'], (3,4))
        R0_rect = np.reshape(rawdata['R0_rect'], (3,3))
        Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3,4))

        data['P0'] = P0
        data['P1'] = P1
        data['P2'] = P2
        data['P3'] = P3
        data['R0_rect'] = R0_rect
        data['Tr_velo_to_cam'] = Tr_velo_to_cam

        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

#This function is used for the baseline (radomized ablation). use test_mode to control whether the ablation percentage is fixed 
def jointly_ablate_images(img1: Image.Image, img2: Image.Image, percentage: float,test_mode) -> (Image.Image, Image.Image):
    # Convert PIL images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2).reshape(img2.shape[0], img2.shape[1], 1)  # Add a third dimension for consistency

    # Check if the height and width dimensions match
    assert arr1.shape[0] == arr2.shape[0] and arr1.shape[1] == arr2.shape[1], "Height and width dimensions must match."

    if not test_mode:
        percentage = random.uniform(0, percentage)

    # Create a mask for each "pixel" (3 channels in img1 and 1 channel in img2)
    total_pixels = arr1.shape[0] * arr1.shape[1]
    num_retain = int(total_pixels * percentage)

    joint_mask = np.concatenate((np.ones(num_retain), np.zeros(total_pixels - num_retain)))
    np.random.shuffle(joint_mask)
    joint_mask = joint_mask.reshape(arr1.shape[0], arr1.shape[1])

    # Apply the mask to each image
    ablated_arr1 = arr1 * joint_mask[:,:,np.newaxis].astype(arr1.dtype)
    ablated_arr2 = arr2 * joint_mask[:,:,np.newaxis].astype(arr2.dtype)

    # Convert ablated arrays back to PIL images
    #ablated_img1 = Image.fromarray(ablated_arr1)
    #ablated_img2 = Image.fromarray(ablated_arr2.squeeze(2).astype(arr2.dtype)/100)
    #ablated_img1.show()
    #ablated_img2.show()
    return ablated_arr1, ablated_arr2.squeeze(2).astype(arr2.dtype)

def randomly_ablate_train(img, ablation_ratio=0.2, ablation_value=0):
    """
    Randomly ablating a random percentage of pixels in the image. This function is for the training purpose.

    Parameters:
    - img: PIL image to be ablated
    - percentage: Fraction of pixels to be retained
    - ablation_value: The value to set the ablated pixels to (default: 0)

    Returns:
    - Ablated PIL image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Generate a random mask for ablation
    percentage = random.uniform(0, ablation_ratio)
    mask = np.random.rand(*img_array.shape[:2]) < 1-percentage

    # Ablate the image
    if len(img_array.shape) == 3:  # if the image has multiple channels (e.g., RGB)
        for channel in range(img_array.shape[2]):
            img_array[mask, channel] = ablation_value
    else:
        img_array[mask] = ablation_value
        #print(np.max(img_array),np.min(img_array))
        
    # Convert back to PIL Image and return
    return img_array

def randomly_ablate_test(img, ablation_ratio=0.1, ablation_value=0):
    """
    Randomly ablating a fixed percentage of pixels in the image. This function if for 

    Parameters:
    - img: PIL image to be ablated
    - percentage: Fraction of pixels to be retained
    - ablation_value: The value to set the ablated pixels to (default: 0)

    Returns:
    - Ablated PIL image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Calculate the number of pixels to ablate
    total_pixels = img_array.shape[0] * img_array.shape[1]
    num_pixels_to_ablate = int((1-ablation_ratio)* total_pixels)

    # Generate random indices to ablate
    indices = np.random.choice(total_pixels, num_pixels_to_ablate, replace=False)
    rows = indices // img_array.shape[1]
    cols = indices % img_array.shape[1]

    # Ablate the image
    if len(img_array.shape) == 3:  # if the image has multiple channels (e.g., RGB)
        for channel in range(img_array.shape[2]):
            img_array[rows, cols, channel] = ablation_value
    else:
        img_array[rows, cols] = ablation_value

    # Convert back to PIL Image and return
    return img_array
def list_files(directory):
    with os.scandir(directory) as entries:
        return [entry.path for entry in entries if entry.is_file()]
class kittidataset(BaseDataset):
    """dataloader for kitti dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot # path for the dataset
        self.use_sne = opt.use_sne
        self.num_labels = 2
        self.use_size = (opt.useWidth, opt.useHeight)
        self.phase = opt.phase
        self.certification_method = opt.certification_method
        if self.use_sne:
            self.sne_model = SNE()

        self.image_list = list_files(os.path.join(self.root,'image_2'))

    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]
        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'image_2', name)), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(os.path.join(useDir, 'depth_u16', name), cv2.IMREAD_ANYDEPTH)
        oriHeight, oriWidth, _ = rgb_image.shape
        label_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'gt_image_2', name[:-10]+'road_'+name[-10:])), cv2.COLOR_BGR2RGB)
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image[:,:,2] > 0] = 1

        if self.phase == "train":
            if self.opt.certification_method == "MMCert":
                rgb_image = randomly_ablate_train(rgb_image, ablation_ratio=self.opt.ablation_ratio_train, ablation_value=0)
                depth_image = randomly_ablate_train(depth_image, ablation_ratio=self.opt.ablation_ratio_train, ablation_value=0)
            else:
                rgb_image,depth_image = jointly_ablate_images(rgb_image, depth_image, percentage = self.opt.ablation_ratio_train,test_mode = False)
        if self.phase == "test":
            if self.opt.certification_method == "MMCert":
                rgb_image = randomly_ablate_test(rgb_image, ablation_ratio=self.opt.ablation_ratio_test1, ablation_value=0)
                depth_image = randomly_ablate_test(depth_image, ablation_ratio=self.opt.ablation_ratio_test2, ablation_value=0)
            else:
                rgb_image,depth_image = jointly_ablate_images(rgb_image, depth_image, percentage = self.opt.ablation_ratio_test,test_mode = True)
        else:              
            print("phase does not exist")

        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)
        
        # another_image will be normal when using SNE, otherwise will be depth
        if self.use_sne:
            calib = kittiCalibInfo(os.path.join(useDir, 'calib', name[:-4]+'.txt'))
            camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)
            normal = self.sne_model(torch.tensor(depth_image.astype(np.float32)/1000), camParam)
            another_image = normal.cpu().numpy()
            another_image = np.transpose(another_image, [1, 2, 0])
            another_image = cv2.resize(another_image, self.use_size)
        else:
            another_image = depth_image.astype(np.float32)/65535
            another_image = cv2.resize(another_image, self.use_size)
            another_image = another_image[:,:,np.newaxis]

        label[label > 0] = 1
        rgb_image = rgb_image.astype(np.float32) / 255

        rgb_image = transforms.ToTensor()(rgb_image)
        another_image = transforms.ToTensor()(another_image)

        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': name, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'kitti'
