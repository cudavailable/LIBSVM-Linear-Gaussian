import os
import csv
import numpy as np
from sklearn import svm
from logger import Logger

def getData(filename, logger):
	""" Get data for svm"""
	"""
	:param filename: data path
	:param logger: log object
	:return : data & labels
	"""
	logger.write("Dataset: \n----------------------------------------\n")

	X = [] # data
	Y = [] # labels

	with open(filename) as f:
		reader = csv.reader(f) # get reader iterator
		head_row = next(reader) # get headline
		logger.write(f"{head_row}\n")
		# print(reader)

		for line in reader:
			logger.write(f"{line}\n") # print watermelon dataset
			X.append(line[7:9])
			Y.append(line[10])

		return X, Y

def linearKernel(X, Y, logger):
	""" linear kernel function """
	""" 
	:param X: data
	:param Y: labels
	:param logger: log object
	:return :
	"""
	logger.write("\nLinear kernel: \n----------------------------------------\n")

	clf = svm.SVC(kernel='linear')
	clf.fit(X, Y)
	logger.write(f"support vectors of linear kernel function: \n{clf.support_vectors_}")
	logger.write(f"\nnumbers of support vectors of each class: \n{clf.n_support_}\n")

def gaussianKernel(X, Y, logger):
	""" gaussian kernel function """
	"""
	:param X: data
	:param Y: labels
	:param logger: log object
	:return :
	"""
	logger.write("\nGaussian kernel: \n----------------------------------------\n")

	clf = svm.SVC(kernel='rbf')
	clf.fit(X, Y)
	logger.write(f"support vectors of gaussian kernel function: \n{clf.support_vectors_}")
	logger.write(f"\nnumbers of support vectors of each class: \n{clf.n_support_}\n")

def main():
	data_path = "./watermelon3.0.csv"
	log_dir = "./log"
	if log_dir is not None and not os.path.exists(log_dir):
		os.mkdir(log_dir)
	logger = Logger(os.path.join(log_dir, "log.txt"))

	X, Y = getData(data_path, logger)
	linearKernel(X, Y, logger)
	gaussianKernel(X, Y, logger)

	logger.close()

if __name__ == "__main__":
	main()