import numpy as np
import os
import glob
from abc import abstractmethod

class DataLoader(object):
	def get_images(self, data_dir):
		files = []
		for ext in ['jpg', 'png', 'jpeg', 'JPG']:
			files.extend(glob.glob(os.path.join(data_dir, '*.{}'.format(ext))))  #.extend 在一个list中添加一个序列的值（not 一个值
		print(files)
		return files


	@abstractmethod
	def load_annotation(self, gt_file):
		print("reimplement by particular data loader")
		pass

