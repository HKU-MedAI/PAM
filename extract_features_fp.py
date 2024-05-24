import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
from math import floor
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.convnext import convnext_small
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
from einops import rearrange, repeat
import cv2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='../Dataset/TCGA-PRAD/patch_4096')
parser.add_argument('--data_slide_dir', type=str, default='../Dataset/TCGA-PRAD/WSI')
parser.add_argument('--csv_path', type=str, default='')
parser.add_argument('--feat_dir', type=str, default='../Dataset/TCGA-PRAD/patch_4096/convnexts_l0l1_512_4096')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, x2, y = dataset[0]
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, batch2, coords) in enumerate(loader):
		with torch.no_grad():
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			import ipdb;ipdb.set_trace()
			batch = batch.to(device, non_blocking=True)
			batch2 = batch2.to(device, non_blocking=True)

			batch = batch.unfold(2, 512, 512).unfold(3, 512, 512)
			batch = rearrange(batch, 'b c p1 p2 w h -> (b p1 p2) c w h')
			
			features = model(batch)
			features2 = model(batch2)

			features = rearrange(features, '(b p1 p2) c -> b p1 p2 c', p1=8, p2=8)
			features = features.cpu().numpy()
			features2 = features2.cpu().numpy()

			coords = np.array(coords)
			asset_dict = {'features': features, 'features2': features2, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

if __name__ == '__main__':

	print('initializing dataset')
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	model = convnext_small(pretrained=True)
	model = model.to(device)

	model.eval()
	slides = os.listdir(args.data_h5_dir + '/patches')
	total = len(slides)

	for bag_candidate_idx in range(total):
		slide_id = slides[bag_candidate_idx].split('.h5')[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")
