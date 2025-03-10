import numpy as np 
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

import pdb
import random
import torch
import time
import cv2


data_path = '../224kfold/'

class PACS(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label)-1)
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			
		

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		
		

		seed = 777

		
		self.domain_list.append(test_domain)
		
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				
				
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label)-1)
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		
		f = open('../files/' + test_domain + '_test_kfold.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)

		
		
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		

	def reset(self, phase, domain_id, transform=None):
		
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]
			

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list
		
		
		
		
		
		
		
		
		

		elif phase == 'ttt':
			self.transform = transform
			len_selfImg = len(self.test_img_list)
			perc = int(0.25*len_selfImg)
			self.img_list = self.test_img_list[:perc]
			self.label_list = self.test_label_list[:perc]
			print('\t \t Total Length of dataset ',len_selfImg )
			print('\t \t Using only this amount of dataset', perc, 'Number',len(self.img_list))
		


		
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		
		image = Image.open(self.img_list[item]).convert('RGB')  
		img_name = self.img_list[item]
		
		
		
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		
		
		return image, label, img_name

	def __len__(self):
		return len(self.img_list)