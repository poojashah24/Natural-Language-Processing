from __future__ import division
from time import localtime, strftime
__author__="Pooja Shah <pbs2124@columbia.edu> "

""" Module imports """
import pdb
import math
import re
import os
from collections import defaultdict

train_file = open('parse_train.dat','r')

''' Dictionary to maintain counts of all terminals (fringe words) '''
word_cnt_map = defaultdict(int)
'''Dictionary to maintain rare words'''
rare_words = dict()

''' Dictionaries to maintain counts of nonterminals, unary rules and binary rules '''
nt_count_map = defaultdict(int)
binaryrule_count_map = defaultdict(int)
unaryrule_count_map = defaultdict(int)

''' Dictionary to maintain values of pi, backpointers and rules for the CKY algorithm '''
pi = defaultdict(float)
bp = defaultdict(int)
rule = {}

''' Regex to find the fringe words in json representing a parse tree '''
fringe_words_regex = '\[[^(\[\])]*,[^(\[\])]*\]'

''' This function finds rare words (<5) in the training file and replaces them with the symbol _RARE_ (using replace_rare_words)'''
def find_and_replace_rare_words(training_file, rare_train_file):
	global train_file
	train_file = open(training_file, 'r')
	rare_file = open(str(rare_train_file), 'w')

	for line in train_file:
		words = re.findall(fringe_words_regex, line)
		for w in words:
			word = w.split(',',1)[1][2:-2]
			word_cnt_map[word] += 1

	for word in word_cnt_map:
		if int(word_cnt_map[word]) < 5:
			rare_words[word] = '_RARE_'

	replace_rare_words(rare_file)

	train_file.close()
	rare_file.close()

''' This function replaces rare words with _RARE_ '''
def replace_rare_words(rare_file):
	train_file.seek(0)

	for line in train_file:
		words = re.findall(fringe_words_regex, line)
		for w in words:
			arr = w.split(',',1)
			w1 = arr[1][2:-2]
			if w1 in rare_words:
				temp = arr[0]+ ", \"_RARE_\"]";
				line = line.replace(w,temp,1)
		rare_file.write(line)

''' This function calculates the number of nonterminals, unary and binary rules '''
def count_events(filename):
	count_file = open(filename,'r')
	for line in  count_file:
		arr = line.split(' ')
		if arr[1] == 'NONTERMINAL':
			nt_count_map[arr[2].strip('\n')] = int(arr[0])
		elif arr[1] == 'UNARYRULE':
			unaryrule_count_map[arr[2], arr[3].strip('\n')] = int(arr[0])
		elif arr[1] == 'BINARYRULE':
			binaryrule_count_map[arr[2],arr[3],arr[4].strip('\n')] = int(arr[0])
	count_file.close()

''' This function computes parameters for binary rules '''
def compute_binary_parameter(x,y1,y2):
	rule_count = binaryrule_count_map[x,y1,y2]
	nt_count = nt_count_map[x]
	return float(rule_count/nt_count)

''' This function computes parameters for unary rules '''
def compute_unary_parameter(x,w):
	rule_count = unaryrule_count_map[x,w]
	nt_count = nt_count_map[x]
	return float(rule_count/nt_count)

''' This function checks if a given word has been seen in the training data '''
def is_unseen(word):
	return False if word in word_cnt_map else True

''' This function read in the test data line by line, and invokes the CKY algorithm for each sentence, generating a parse tree '''
def CKYDriver(test_file, prediction_file):
	test = open(test_file,'r')
	prediction = open(prediction_file,'w')

	for line in test:
		res = CKY(line.strip('\n'))
		prediction.write(res[1]+'\n')

	test.close()
	prediction.close()

''' This function is the implementation of the CKY algorithm '''
def CKY(sentence):

	''' initialization '''
	words = sentence.split(' ')
	n = len(words)

	''' initialize pi values and backpointers '''
	''' the pi value is non-zero and equal to the probability of the unary rule if the rule X->x(i) exists, else it is 0 '''

	i = 0
	''' ierate over the words in a sentence and all the nonterminals, initializing pi values for each '''
	for word in words:
		i += 1
		for nt in nt_count_map:
			w = '_RARE_' if word in rare_words or is_unseen(word) else word
			pi[i,i,nt] = compute_unary_parameter(nt, w)
			rule[i,i,nt] = [nt, w]
			bp[i,i,nt] = 0

	''' algorithm '''
	for l in range(1, n):
		for i in range(1, (n-l)+1):
			j = i + l

			''' iterate over all the nonterminals '''
			for nt in nt_count_map:
				max_prob = 0.0
				maxbp = -1
				maxrule = []
				maxbp = None

				for binaryrule in binaryrule_count_map:
					if binaryrule[0] == nt:
						x = binaryrule[0]
						y = binaryrule[1]
						z = binaryrule[2]

						for s in range(i,j):
							if pi[i,s,y] and pi[s+1,j,z]:
								temp = compute_binary_parameter(x,y,z) * pi[i,s,y] * pi[s+1,j,z]
								''' find the rule (X->YZ) and the split point that gives the maximum probability '''
								if temp > max_prob:
									max_prob = temp
									maxbp = s
									maxrule = [nt,y,z]

				''' save the rule X->YZ and the split point that gave the maximum probability '''
				pi[i,j,nt] = max_prob
				bp[i,j,nt] = maxbp
				rule[i,j,nt] = maxrule

	''' output'''
	global json

	''' use the backpointers to retrieve the parse tree corresponding to the sentence '''
	''' the else clause is added for sentences that are simply fragments '''
	if pi[1,n,'S']:
		json = ''
		print_tree(1,n,'S',words)
		return pi[1,n,'S'],json
	else:
		max_prob = 0.0
		max_nt = ''
		for nt in nt_count_map:
			temp = pi[1,n,nt]
			if temp > max_prob:
				max_prob = temp
				max_nt = nt
		json = ''
		print_tree(1,n,max_nt,words)
		return max_prob,json


''' This function retrieves the parse tree from the backpointers '''
json = ''
def print_tree(i,j,nt,words):
	global json
	if i!=0 and j!=0:
		''' add the LHS of the rule to the JSON '''
		json = json.strip('\n') + "[\"" + rule[i,j,nt][0].strip('\n') + "\","

		''' find the split point '''
		s = bp[i,j,nt]
		arr = rule[i,j,nt]

		''' recurse in the left subtree '''
		print_tree(i,s,arr[1],words)
		if(len(arr) == 3):
			json = json.strip('\n') + ","

			''' recurse in the right subtree '''
			print_tree(s+1,j,arr[2],words)
			json = json.strip('\n') +  "]"
		else:
			json = json.strip('\n') + "\"" + rule[i,j,nt][1].strip('\n') + "\"]"
	else:
		return


def run_program(training_file, rare_train_file, counts_file, prediction_file, test_file):
	print('start time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))
	train_file = training_file
	find_and_replace_rare_words(train_file, rare_train_file)
	count_events_cmd = "python count_cfg_freq.py " + rare_train_file + " > " + counts_file
	os.system(count_events_cmd)
	count_events(counts_file)
	CKYDriver(test_file,prediction_file)
	print('end time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))




