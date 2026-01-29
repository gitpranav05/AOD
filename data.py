# import os
# import torch
# from PIL import Image
# import glob
# import random


# class HazeDataset(torch.utils.data.Dataset):
#     def __init__(self, ori_root, haze_root, transforms):
#         self.haze_root = haze_root
#         self.ori_root = ori_root
#         self.image_name_list = glob.glob(os.path.join(self.haze_root, '*.jpg'))
#         self.matching_dict = {}
#         self.file_list = []
#         self.get_image_pair_list()
#         self.transforms = transforms
#         print("Total data examples:", len(self.file_list))

#     def __getitem__(self, item):
#         """
#         :param item:
#         :return: haze_img, ori_img
#         """
#         ori_image_name, haze_image_name = self.file_list[item]
#         ori_image = self.transforms(Image.open(ori_image_name))
#         haze_image = self.transforms(Image.open(haze_image_name))
#         return ori_image, haze_image

#     def __len__(self):
#         return len(self.file_list)

#     def get_image_pair_list(self):
#         for image in self.image_name_list:
#             image = image.split("/")[-1]
#             key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
#             if key in self.matching_dict.keys():
#                 self.matching_dict[key].append(image)
#             else:
#                 self.matching_dict[key] = []
#                 self.matching_dict[key].append(image)

#         for key in list(self.matching_dict.keys()):
#             for hazy_image in self.matching_dict[key]:
#                 self.file_list.append([os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image)])

#         random.shuffle(self.file_list)


import os
import torch
from PIL import Image
import glob
import random


class HazeDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, haze_root, transforms):
        self.haze_root = haze_root
        self.ori_root = ori_root
        self.transforms = transforms

        self.file_list = []
        self.get_image_pair_list()

        print("Total data examples:", len(self.file_list))

    def __getitem__(self, index):
        ori_path, haze_path = self.file_list[index]

        ori_image = Image.open(ori_path).convert("RGB")
        haze_image = Image.open(haze_path).convert("RGB")

        if self.transforms:
            ori_image = self.transforms(ori_image)
            haze_image = self.transforms(haze_image)

        # IMPORTANT ORDER (matches train.py)
        return ori_image, haze_image

    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        haze_images = sorted(glob.glob(os.path.join(self.haze_root, '*')))
        ori_images = sorted(glob.glob(os.path.join(self.ori_root, '*')))

        haze_dict = {os.path.basename(p): p for p in haze_images}
        ori_dict = {os.path.basename(p): p for p in ori_images}

        common_names = set(haze_dict.keys()).intersection(set(ori_dict.keys()))

        for name in common_names:
            self.file_list.append([ori_dict[name], haze_dict[name]])

        random.shuffle(self.file_list)

