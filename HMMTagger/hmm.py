__author__="Pooja Shah <pbs2124@columbia.edu> "

""" Module imports """
import pdb
import math
import re
import os

f = open('ner_dev.dat', 'r')
train_f = open('ner_train.dat', 'r')
test_f = open('ner_dev.dat', 'r')

""" Regular expressions for pattern matching for numerals, capital words and words with only capitals and periods """
numeral_regex = re.compile('^-?[0-9]+$')
capitals_regex = re.compile('^[A-Z]+$')
abbreviations_regex = re.compile('^([A-Z]+\.)+$')

""" Dictionaries for storing n-gram counts, rare words, capitals, numerals, abbreviations, pi values and backpointers """
d = dict()
gram_dict = dict()
words_dict = dict()
rare_words = dict()
numerals = dict()
capitals = dict()
abbreviations = dict()
pi_map = dict()
pi_map['-1'+'*'+'*'] = 1
bp = dict()
y = dict()
prob_map = dict()
tag_cnt_dict = dict()

log = False
groups = False

""" Tag definitions """
tags = ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'O']
tag_arr = dict()
start = ['*']


""" This function populates 2 dictionaries - 
			1) a map of how many times a word is tagged to a particular tag value.
			2) a map of how many times a tag appears in the training set """

def populate_map_and_counts():
	for t in tags:
		tag_cnt_dict[t] = 0

	for line in f:
	    if line.find('WORDTAG') != -1:
	        arr = line.split(' ')
	        arr.remove('WORDTAG')

	        s = arr[2].strip('\n') + '_' + str(arr[1])
	        d[s] = arr[0]

	        tag_cnt_dict[arr[1]] = tag_cnt_dict[arr[1]] + int(arr[0])



""" This function computes emission parameters e(x|y) = Count(y ~ x) / Count(y)
	For a given tag and word, the emission parameters is calculated. If the word is rare, it is assigned the emission parameters of 
	a rare word for the given tag. Similarly, it is checked for being a numeric value, all capitals or an abbreviation. 
	"""

def get_emission_parameters (tag, word):
	tag_cnt = 0
	str = word + '_' + tag
	if log:
		print ('emission:%s' %str)

	try:
		cnt = d[str]
		if log:
			print ('cnt: %s' %cnt)
	except KeyError:
		
		if groups and isnumeral(word):
				try:
					cnt = d['_NUM_' + '_' + tag]
				except:
					return 0.0
		elif groups and isallcaps(word):
				try:
					cnt = d['_ALLCAPS_' + '_' + tag]
				except:
					return 0.0
		elif groups and isabbreviation(word):
				try:
					cnt = d['_ABBRV_' + '_' + tag]
				except:
					return 0.0
		elif word in rare_words or is_unseen(word):
			try:
				cnt = d['_RARE_' + '_' + tag]
			except:
				return 0.0
		else:
			cnt = 0.0
		
	tag_cnt = tag_cnt_dict[tag]

	if tag_cnt != 0:
		cnt = float(cnt) / tag_cnt
	else:
		cnt = 0

	if log:
		print ('cnt: %s' %cnt)

	return cnt

""" The next three functions are utility functions to check if a word is numeric, all capitals or an abbreviation """
def isnumeral(word):
	return numeral_regex.match(word)

def isallcaps(word):
	return capitals_regex.match(word)

def isabbreviation(word):
	return abbreviations_regex.match(word)



""" This function identifies rare words and replaces them by the word _RARE_ 
It also categorizes words into other groups (if the groups option is set) and replaces these words appropriately """
def find_rare_words (filename, groups):
	#print ('filename: %s' %filename)
	rare_f = open(str(filename), 'w')

	for line in train_f:
		arr = line.split(' ')
		wordtag = arr[0].strip('\n')
		if wordtag in words_dict:
		    words_dict[wordtag] = int(words_dict[wordtag]) + 1
		else:
		    words_dict[wordtag] = 1

	for x in words_dict:
		if int(words_dict[x]) < 5:
			if groups and isnumeral(x):
				numerals[x] = '_NUM_'
			elif groups and isallcaps(x):
				capitals[x] = '_ALLCAPS_'
			elif groups and isabbreviation(x):
				abbreviations[x] = '_ABBRV_'
			else:
				rare_words[x] = '_RARE_'

	replace_by_rare(rare_f)

	train_f.close()
	rare_f.close()

""" This function replaces rare words by their appropriate category """
def replace_by_rare (rare_file):
	train_f.seek(0)
	for line in train_f:
		arr = line.split(' ')
		w = arr[0].strip('\n')
		if w in rare_words:
			rare_file.write(line.replace(w, '_RARE_', 1))
		elif w in numerals:
			rare_file.write(line.replace(w, '_NUM_', 1))
		elif w in capitals:
			rare_file.write(line.replace(w, '_ALLCAPS_', 1))
		elif w in abbreviations:
			rare_file.write(line.replace(w, '_ABBRV_', 1))
		else:
			rare_file.write(line)

""" This function checks if a word has never been seen before """
def is_unseen(word):
	for tag in tags:
		str1 = word + '_' + tag
		if d.has_key(str1):
			return False
	return True

""" This function returns the most likely tag for a word based on emission parameters (Question 4) 
It checks the emission parameters for the word with all possible tags, and return the tag with highest value """
def get_predicted_tag (word):
	tag = ''
	emission_param = 0.0
	for t in tags:
		res = get_emission_parameters(t, word)
		if res > emission_param:
			emission_param = res
			tag = t

	return tag, emission_param 

""" This functions acts as a named entity tagger, tagging all words in an input test file (Question 4)
It reads the test data, and for each word gets the predicted tag. The word,its predicted tag and the log probability 
for each prediction is printed to an output file """
def predict_tags(prediction_file):
	test = open('ner_dev.dat','r')
	results = open(prediction_file, 'w')
	log_prob = 0.0
	for line in test:
		res = get_predicted_tag(line.strip('\n'))
		if not line.strip('\n'):
			resline = '\n'
		else:
			if float(res[1] == 0.0):
				log_prob = 0.0
			else:
				log_prob = math.log(res[1])

			resline = line.strip('\n') + ' ' + str(res[0]) + ' ' + str(log_prob) + '\n'
		results.write(resline)
	results.close()

""" This function populates a dictionary with n-gram counts for each n-gram in a counts file """
def populate_ngram_counts():
	f.seek(0)
	for line in f:
		    if line.find('GRAM') != -1:
		        arr = line.split(' ')
		        if arr[1] == '1-GRAM':
		        	gram_dict[arr[2].strip('\n')] = arr[0]
		        elif arr[1] == '2-GRAM':
		        	gram_dict[(arr[2] + '_' + arr[3]).strip('\n')] = arr[0]
		        elif arr[1] == '3-GRAM':
		        	gram_dict[(arr[2] + '_' + arr[3] + '_' + arr[4]).strip('\n')] = arr[0]

""" This function computes parameters q(yi|yi-1, yi-2) = Count(yi-2, yi-1, yi) / Count(yi-2, yi-1) for a given trigram 
				yi-2 yi-1 yi """
def compute_trigram_params(yi2, yi1, yi):
	if not yi2:
		yi2 = '*'
	if not yi1:
		yi1 = '*'
	if not yi:
		yi = 'STOP'

	trigram = yi2 + '_' + yi1 + '_' + yi
	bigram = yi2 + '_' + yi1

	try:
		trigram_count =  float(gram_dict[trigram.strip('\n')])
	except:
		trigram_count = 0

	try:
		bigram_count = float(gram_dict[bigram])
	except:
		bigram_count = 0

	if bigram_count != 0:
		res = trigram_count / bigram_count
	else:
		res = 0
	
	return res

"""A function that reads in lines of state trigrams yi-2, yi-1 yi (separated by space) and prints the 
	log probability for each trigram to the result file. """
def compute_trigram_prob(trigramfile):
	result = open(trigramfile, 'w')
	resline = ''
	f.seek(0)
	for line in f:
		if line.find('3-GRAM') != -1:
			arr = line.split(' ')
			res = compute_trigram_params(arr[2], arr[3], arr[4])
			if res != 0:
				res = math.log(res)
			resline = arr[2] + ' ' + arr[3] + ' ' + arr[4].strip('\n') + ' ' + str(res) + '\n'
			result.write(resline)
	result.close()


""" This function is a driver function for the viterbi algorithms. It takes in a sentence,
initializes the data structures for the viterbi algorithm, and then invokes the viterbi algorithm for the sentence 
Particularly it executes the outer loop iterating over all the words in the sentence """
def viterbi_driver(sentence):
	res = 0
	words = sentence.split(' ')
	n = len(words)

	""" Initialize the tags for each possible location """
	tag_arr[-2] = start
	tag_arr[-1] = start
	
	for k in range(0, n):
		tag_arr[k] = tags

	""" Initialize data structures for the sentence """
	pi_map.clear()
	pi_map['-1'+'_'+'*'+'_'+'*'] = 1
	bp.clear()

	""" For each word invoke the inner loops iterating over 'u','v' and 'w' values """
	for k in range(0,n):
		res = viterbi(k, '*', '*', words)

	""" Reconstruct the tags which led to the maximum probability tag sequence"""
	maxval = -10000
	tag = ''
	u = ''
	v = ''
	prob = 0.0
	stop_param = 0.0

	'''for tag in pi_map:
			print ('tag: %s, prob: %s' %(tag, pi_map[tag]))'''

	for i in pi_map:
		arr = i.split('_')
		if int(arr[0]) == n-1:
			if pi_map[i] != -1000000.0:
				stop_param = compute_trigram_params(arr[1], arr[2], 'STOP')
				prob = pi_map[i] * stop_param
				if prob > maxval:
					maxval = prob
					u = arr[1]
					v = arr[2]

	y.clear()
	prob_map.clear()
	y[words[n-2]] = u
	y[words[n-1]] = v
	
	prob_map[words[n-1]] = pi_map[str(n-1) + '_' + u + '_' + v]

	for z in range(n-3,-1,-1):
		y[words[z]] = bp[str(z+2) + y[words[z+1]] + y[words[z+2]]]
		prob_map[words[z+1]] = pi_map[str(z+1) + '_' + y[words[z]] + '_' + y[words[z+1]]]
	return res
		

""" This function executes the inner loops of the viterbi algoithm where values of u, v and w are iterated over to find 
the maximum value of w for all combinations of u and v tags. It also records the tags leading to the maximum value""" 
def viterbi(k, u, v, words):
	res = 0.0
	maxval = -1000000.0
	wval = 'O'
	tri_par = 0.0
	emi_par = 0.0

	for u in tag_arr[k-1]:
		for v in tag_arr[k]:
			maxval = -1000000.0
			res = 0.0
			for w in tag_arr[k-2]:
					emi_par = get_emission_parameters(v, words[k])
					res = pi_map[str(k-1) + '_' + w + '_' + u]
					tri_par = compute_trigram_params(w, u, v)
					res = res * tri_par * emi_par

					if res != 0 and res != -0 and emi_par != 0 and tri_par != -1000:
						if res > maxval:
							maxval = res
							wval = w


			pi_map[str(k) + '_' + u + '_' + v] = maxval
			bp[str(k) + u + v] = wval
			prob_map[words[k]] = maxval
	return maxval


""" This function reads in the test data a line at a time, constructs sentences and passes it to the viterbi driver function
It also writes the output tags and probability to the results file """
def predict_using_viterbi(prediction_file):
	input = ''
	resline = ''
	res = 1.0
	results = open(prediction_file, 'w')
	test_f.seek(0)
	for line in test_f:
		line = line.strip('\n').strip('\n').strip(' ')
		if line:
			input += ' ' + line.strip('\n')
		else:
			res = viterbi_driver(input)
			arr = input.split(' ')
			res = 1.0
			for x in arr:	
				if x.strip(' '):
					res = prob_map[x.strip('\n')]
					resline = x + ' ' + y[x.strip('\n')] + ' ' + str(res) + '\n'
					results.write(resline)
			input = ''
			results.write('\n')

	results.close()


""" This function is the entry point to execute the different prediction algorithms. It also performs pre-processing actions """
def run_program(train_rare_file, enablegroup, counts_file, prediction_file, trigram_file, prediction_file_viterbi):
	global groups
	global f
	groups = enablegroup
	find_rare_words(train_rare_file, groups)
	count_freqs_cmd = "python count_freqs.py " + train_rare_file + " > " + counts_file
	os.system(count_freqs_cmd)
	f = open(counts_file, 'r')
	populate_map_and_counts()
	predict_tags(prediction_file)
	populate_ngram_counts()
	compute_trigram_prob(trigram_file)
	predict_using_viterbi(prediction_file_viterbi)











