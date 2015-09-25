__author__="Pooja Shah <pbs2124@columbia.edu> "

""" Module imports """
import pdb
import math
import re
import os

train_file = open('parse_train.dat','r')


def find_and_replace_rare_words(filename):
	rare_file = open(str(filename), 'w')

	for line in train_file:
		arr = re.findall('\[.*,.*\]', line)
		for x in arr:
			print ('x:%s' %x)

	train_file.close()
	rare_file.close()


find_and_replace_rare_words("parse_train.dat")


