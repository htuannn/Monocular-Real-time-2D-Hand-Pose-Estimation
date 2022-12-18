import numpy as np
import os
import json
import glob
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from natsort import natsorted

from sklearn.model_selection import train_test_split

from utils import (
	projectPoints,
	N_JOINTS,
	gen_heatmap,
	RAW_IMG_SIZE,
	MODEL_INPUT_SIZE,
	MODEL_OUTPUT_SIZE,
	HEATMAP_SIGMA,
	vector_to_heatmaps,
)

class FreiHand(Dataset):
	def __init__(self, config, n_sample=1, set_type="train"):
			##initialize path to the image folder
			self.device=config["device"]
			self.n_sample=n_sample
			self.imgs_path=os.path.join(config["data_dir"],"training/rgb")
			self.imgs_list=natsorted(os.listdir(self.imgs_path))

			##initialize path to the files with label
			path_K_matrices= os.path.join(config["data_dir"],"training_K.json")
			with open(path_K_matrices , "r") as f:
				self.K_matrices=np.array(json.load(f))
				
			path_xyz_coor= os.path.join(config["data_dir"],"training_xyz.json")
			with open(path_xyz_coor , "r") as f:
				self.xyz_coor=np.array(json.load(f))

			##split set
			"""
			train:val:test=8:1:1
			"""
			#transform coord (raw_img_size * raw_img_size) -> (1,1)
			self.keypoints = [projectPoints(x,y)/ RAW_IMG_SIZE for (x,y) in list(zip(self.xyz_coor,self.K_matrices))] * self.n_sample

			train_list, rem_list, train_key, rem_key= train_test_split(self.imgs_list[: len(self.keypoints)], self.keypoints , train_size=0.8, random_state=18)
			val_list, test_list, val_key, test_key= train_test_split(rem_list, rem_key, train_size=0.5, random_state=18)

			if set_type == "train":
				self.imgs_list=train_list
				self.keypoints=train_key
			elif set_type == "val":
				self.imgs_list=val_list
				self.keypoints=val_key
			else:
				self.imgs_list=test_list
				self.keypoints=test_key

			self.image_transform = transforms.Compose([
				transforms.Resize(MODEL_INPUT_SIZE),
				transforms.ToTensor(),
				])

	def __len__(self):
		"""
		returns length of the dataset
		"""
		return len(self.imgs_list)

	def __getitem__(self,idx):
		img_name= self.imgs_list[idx]
		img_raw=Image.open(os.path.join(self.imgs_path,img_name))
		img_preprocessed=self.image_transform(img_raw)
		img_raw_transfrom=transforms.ToTensor()(img_raw)

		keypoints=self.keypoints[idx]
		heatmaps= vector_to_heatmaps(keypoints)	
		keypoints= torch.from_numpy(keypoints)
		heatmaps=torch.from_numpy(heatmaps)

		return {
		"img_name": img_name,
		"image": img_preprocessed,
		"heatmaps": heatmaps,
		"keypoints": keypoints,
		"image_raw": img_raw_transfrom,
		}
