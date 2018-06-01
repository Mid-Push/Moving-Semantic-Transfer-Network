import numpy
import os
import sys
import util
from urlparse import urljoin
import gzip
import struct
import operator
import numpy as np
#from preprocessing import preprocessing
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
class MNIST:
	base_url = 'http://yann.lecun.com/exdb/mnist/'

    	data_files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz',
            }
	def __init__(self,path=None,shuffle=True,output_size=[28,28],output_channel=1,split='train',select=[]):
		self.image_shape=(28,28,1)
		self.label_shape=()	
		self.path=path
		self.shuffle=shuffle
		self.output_size=output_size
		self.output_channel=output_channel
		self.split=split
		self.select=select
		self.download()
		self.pointer=0
		self.load_dataset()
	def download(self):
		data_dir = self.path
        	if not os.path.exists(data_dir):
            		os.mkdir(data_dir)
        	for filename in self.data_files.values():
            		path = self.path+'/'+filename
            		if not os.path.exists(path):
                		url = urljoin(self.base_url, filename)
                		util.maybe_download(url, path)
	def _read_datafile(self, path, expected_dims):
        	base_magic_num = 2048
        	with gzip.GzipFile(path) as f:
        		magic_num = struct.unpack('>I', f.read(4))[0]
        		expected_magic_num = base_magic_num + expected_dims
        		if magic_num != expected_magic_num:
        	        	raise ValueError('Incorrect MNIST magic number (expected '
        	                         '{}, got {})'
        	                         .format(expected_magic_num, magic_num))
        	    	dims = struct.unpack('>' + 'I' * expected_dims,
        	                         f.read(4 * expected_dims))
        	    	buf = f.read(reduce(operator.mul, dims))
        	    	data = np.frombuffer(buf, dtype=np.uint8)
        	    	data = data.reshape(*dims)
        	    	return data
        def shuffle_data(self):
		images = self.images[:]
        	labels = self.labels[:]
        	self.images = []
        	self.labels = []

        	idx = np.random.permutation(len(labels))
        	for i in idx:
            		self.images.append(images[i])
            		self.labels.append(labels[i])
	def load_dataset(self):
		abspaths = {name: self.path+'/'+path
                	for name, path in self.data_files.items()}
		if self.split=='train':
			self.images = self._read_images(abspaths['train_images'])
        		self.labels = self._read_labels(abspaths['train_labels'])
        	elif self.split=='test':
			self.images = self._read_images(abspaths['test_images'])
        		self.labels = self._read_labels(abspaths['test_labels'])
		if len(self.select)!=0:
			self.images=self.images[self.select]
			self.labels=self.labels[self.select]
	
	def reset_pointer(self):
		self.pointer=0
		if self.shuffle:
			self.shuffle_data()	

	def class_next_batch(self,num_per_class):
		batch_size=10*num_per_class
		classpaths=[]
		ids=[]
		for i in xrange(10):
			classpaths.append([])
		for j in xrange(len(self.labels)):
			label=self.labels[j]
			classpaths[label].append(j)
	        for i in xrange(10):
			ids+=np.random.choice(classpaths[i],size=num_per_class,replace=False).tolist()
		selfimages=np.array(self.images)
		selflabels=np.array(self.labels)
		return np.array(selfimages[ids]),get_one_hot(selflabels[ids],10)

	def next_batch(self,batch_size):
		if self.pointer+batch_size>=len(self.labels):
			self.reset_pointer()
		images=self.images[self.pointer:(self.pointer+batch_size)]
		labels=self.labels[self.pointer:(self.pointer+batch_size)]
		self.pointer+=batch_size
		return np.array(images),get_one_hot(labels,10)	
	def _read_images(self, path):
        	return (self._read_datafile(path, 3)
                .astype(np.float32)
                .reshape(-1, 28, 28, 1)
                /255.0)

    	def _read_labels(self, path):
        	return self._read_datafile(path, 1)

def main():
	mnist=MNIST(path='data/mnist')
	a,b=mnist.next_batch(1)
	print a
	print b

if __name__=='__main__':
	main()
