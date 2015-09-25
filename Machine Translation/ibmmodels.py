from __future__ import division
from __future__ import print_function
from time import localtime, strftime
__author__="Pooja Shah <pbs2124@columbia.edu> "

""" Module imports """
import pdb
import math
import re
import os
import operator
from collections import defaultdict

''' this is a map to keep count of how many foreign words can be aligned to an english word'''
word_cnt_map = defaultdict(int)

''' this is a map that stores all the foreign words that can be assigned to an english word'''
word_map = dict()

''' this is a map that stores the t parameters i.e. t(f|e) '''
t = defaultdict(float)

''' this is a map that stores the q parameters i.e. q(j|i,l,m)'''
q = defaultdict(float)

''' this stores the c values in the em algorithm'''
c = defaultdict(float)

'''this dictionary stores all the l/m pairs that are seen in the corpus'''
lmpairs = dict()

lines_en = []
lines_de = []
lines_dev_words = []

dev_words = defaultdict(str)

''' this is a driver question for finding the top 10 foreign words for a given english word (as per question 4) '''
def find_foreign_words(foreign_words_filename):
	with open("devwords.txt") as dev_file:
		lines_dev_words = dev_file.readlines()

	for i in range(0,len(lines_dev_words)):
		line = lines_dev_words[i].strip('\n')
		dev_words[line] = line
	find_largest_t(dev_words, foreign_words_filename)

''' this function contains the loggic for finding the top 10 foreign words for a given english word (question 4). to make it efficient, we parse the t parameters only once.
since dictionaries are used the order of the input words doesn't match the order of the output. However, the output contains each input word with the top 10 translations 
(in descending order of "t" parameter value) '''
def find_largest_t(dev_words, foreign_words_filename):
	global t

	output_file = open(foreign_words_filename, 'w')

	top_words = dict()
	count = defaultdict(int)
	for w in dev_words:
		count[w] = 0
		top_words[w] = dict()

	foreign_words = defaultdict(float)
	for d,e in t.items():
		if d[1] in dev_words:
			foreign_words[d[1],d[0]] = t[d[0],d[1]]

	sorted_words = sorted(foreign_words.items(), key=operator.itemgetter(1), reverse=True)
	for w in sorted_words:
		e = w[0][0]
		d = w[0][1]
		if count[e] < 10:
			top_words[e][d] = w[1]
			count[e] += 1
		
	for w in dev_words:
		output_file.write(w + '\n')
		sorted_top_w = sorted(top_words[w].items(), key=operator.itemgetter(1), reverse=True)
		output_file.write('[')
		i=0
		for d in sorted_top_w:
			output_file.write('(\'%s\', %s)' %(d[0],d[1]))
			if i < 9:
				output_file.write(', ')
			i += 1
		output_file.write(']\n\n')

''' this function is the driver function to recover alignments between pairs of English/German sentences'''
def find_alignments(model, output_file_name):
	print('Started finding alignments using IBM Model %s at %s' %(model,strftime("%Y-%m-%d %H:%M:%S", localtime())))
	output_file = open(output_file_name, 'w')
	for s in range(0,20):
		line_en = lines_en[s].strip('\n')
		line_de = lines_de[s].strip('\n')
		output_file.write(line_en + '\n')
		output_file.write(line_de + '\n')
		words_en = line_en.strip('\n').split(' ')
		words_de = line_de.strip('\n').split(' ')
		if model == '1':
			alignments = find_best_alignment_model1(words_en, words_de)
		elif model == '2':
			alignments = find_best_alignment_model2(words_en, words_de)
		output_file.write(str(alignments))
		output_file.write('\n\n')
	output_file.close()

''' this function contains the logic to recover alignments based on IBM Model 1. For each word in the German sentence it chooses the alignment that maximizes the "t" 
parameter (i.e. it considers all alignments positions in the English sentence)'''
def find_best_alignment_model1(words_en, words_de):
	global t
	alignments = []
	for i in range(0,len(words_de)):
		max_prob = 0.0
		max_a = -2;
		for j in range(-1, len(words_en)):
			if j == -1:
				en_word = 'NULL'
			else:
				en_word = words_en[j]

			if t[words_de[i],en_word] > max_prob:
				max_prob = t[words_de[i],en_word]
				max_a = j+1
		alignments.append(max_a)
	return alignments

''' this function contains the logic to recover alignments based on IBM Model 2. For each word in the German sentence it chooses the alignment that maximizes the product of the 
"t" and "q" parameter (i.e. it considers all alignments positions in the English sentence)'''
def find_best_alignment_model2(words_en, words_de):
	global t
	global q
	alignments = []
	l = len(words_en)
	m = len(words_de)

	for i in range(0,len(words_de)):
		max_prob = 0.0
		max_a = -2;
		for j in range(-1, len(words_en)):
			if j == -1:
				en_word = 'NULL'
			else:
				en_word = words_en[j]

			prob = t[words_de[i],en_word] * q[j+1, i+1, l, m]
			if prob > max_prob:
				max_prob = prob
				max_a = j+1
		alignments.append(max_a)
	return alignments

''' this function is the driver function that finds the english sentences with the highest scoring alignment for all the given german sentences'''
def find_translations(german_file_name, scrambled_file_name, unscrambled_file_name):
	global t
	global q

	with open(german_file_name) as de_file:
		lines_de_correct = de_file.readlines()

	with open(scrambled_file_name) as scrambled_en_file:
		lines_en_scrambled = scrambled_en_file.readlines()	

	unscrambled_file = open(unscrambled_file_name,'w')

	''' iterate over all the german sentences'''
	for line_de in lines_de_correct:
		line_de = line_de.strip('\n')
		translation = find_translation(line_de, lines_en_scrambled)
		unscrambled_file.write(translation)
	unscrambled_file.close()

''' this function contains the logic to find the english sentence with the highest scoring alignment for a given german sentence.
For each German sentence, it iterates through all the English sentences, and for each word in the German sentence checks for all the possible alignments.
Finally, it finds the sentence with the highest scoring alignment'''

def find_translation(line_de, lines_en_scrambled):
	global t
	global q

	words_de = line_de.split(' ')
	m = len(words_de)
	max_sentence = ''

	max_sentence_prob = -100000000
	max_sentence = ''

	''' iterate over all the english sentences in the corpus'''
	for line_en in lines_en_scrambled:
		words_en = line_en.split(' ')
		l = len(words_en)

		sentence_prob = 0.0
		''' iterate over each position in the German sentence'''
		for i in range(1,m+1):
			max_prob = -1000000
			''' iterate over all possible alignments in the English sentence for the word in the German sentence'''
			for j in range(0,l+1):
				if j == 0:
					prob = q[j,i,l,m] * t[words_de[i-1],'NULL']
				else:
					prob = q[j,i,l,m] * t[words_de[i-1], words_en[j-1]]
				if prob != 0:
						prob = math.log(prob)
				else:
					prob = -1000000
				''' choose the max prob alignment'''
				if prob > max_prob:
					max_prob = prob
			sentence_prob = sentence_prob + max_prob

		''' choose the max scoring sentence''' 
		if sentence_prob > max_sentence_prob:
			max_sentence_prob = sentence_prob
			max_sentence = line_en

	return max_sentence


''' this function initializes the "q" parameters for IBM Model 2 before executing the EM algorithm'''
def initialize_q_params():
	global q
	global lmpairs
	for entry in lmpairs:
		l = entry[0]
		m = entry[1]
		for i in range(1, m+1):
			for j in range(0,l+1):
				q[j,i,l,m] = 1/l+1


''' this function implements the EM algorithm for IBM Model 1'''
def em_algorithm_model1():
	global t

	''' run 5 iterations'''
	for s in xrange(1,6):
		c = defaultdict(float)
		cprime = defaultdict(float)

		''' iterate over each sentence pair'''
		for k in xrange(0, len(lines_en)):
			words_de = lines_de[k].strip('\n').split(' ')
			words_en = lines_en[k].strip('\n').split(' ')
			m = len(words_de)
			l = len(words_en)

			''' iterate over each position in the German sentence'''
			for i in xrange(0,m):
				summation = calculate_summation(i, words_de, words_en)
				delta = calculate_delta(k, i, -1, words_de, words_en, summation)
				c['NULL', words_de[i]] += delta
				cprime['NULL'] += delta

				''' iterate over each position in the English sentence'''
				for j in xrange(0,l):
					delta = calculate_delta(k, i, j, words_de, words_en, summation)
					c[words_en[j], words_de[i]] += delta
					cprime[words_en[j]] += delta
		
		''' re-estimate model parameters'''
		for entry in c:
			e = entry[0]
			f = entry[1]
			t[f,e] = c[entry] / cprime[e]
		

''' this function implements the EM algorithm for IBM Model 2'''
def em_algorithm_model2():
	global t
	global q
	''' run 5 iterations'''
	for s in xrange(1,6):
		c = defaultdict(float)
		cprime = defaultdict(float)
		c_q = defaultdict(float)
		c_q_prime = defaultdict(float)

		''' iterate over each sentence pair'''
		for k in xrange(0, len(lines_en)):
			words_de = lines_de[k].strip('\n').split(' ')
			words_en = lines_en[k].strip('\n').split(' ')
			m = len(words_de)
			l = len(words_en)

			''' iterate over each position in the German sentence'''
			for i in xrange(0,m):
				summation = calculate_summation_model2(i, words_de, words_en)

				delta = calculate_delta_model2(k,i,-1, words_de, words_en, summation)
				c['NULL', words_de[i]] += delta
				cprime['NULL'] += delta
				c_q[0, i+1, l, m] += delta
				c_q_prime[i+1, l, m] += delta

				''' iterate over each position in the English sentence'''
				for j in xrange(0,l):
					delta = calculate_delta_model2(k,i,j, words_de, words_en,summation)
					c[words_en[j], words_de[i]] += delta
					cprime[words_en[j]] += delta
					c_q[j+1, i+1, l, m] += delta
					c_q_prime[i+1, l, m] += delta

		''' re-estimate model parameters'''
		for entry in c:
			e = entry[0]
			f = entry[1]
			t[f,e] = c[entry] / cprime[e]

		for entry in c_q:
			j = entry[0]
			i = entry[1]
			l = entry[2]
			m = entry[3]
			q[j,i,l,m] = c_q[entry] / c_q_prime[i,l,m]


def calculate_summation(i,words_de,words_en):
	global t
	summation = 0;
	for w in words_en:
		summation += t[words_de[i], w]
	summation += t[words_de[i], 'NULL']

	return summation


''' this function calculates the delta values for IBM Model 1'''
def calculate_delta(k,i,j,words_de,words_en,summation):
	global t
	if j == -1:
		delta = t[words_de[i], 'NULL'] / summation
	else:
		delta = t[words_de[i], words_en[j]] / summation
	return delta

def calculate_summation_model2(i, words_de, words_en):
	global t
	global q

	l = len(words_en)
	m = len(words_de)

	#calculate the denominator term
	summation = 0;
	for k in range(1, l+1):
		summation += q[k,i+1,l,m] * t[words_de[i], words_en[k-1]]
	summation += q[0,i+1,l,m] * t[words_de[i], 'NULL']
	return summation


''' this function calculates the delta values for IBM Model 2'''
def calculate_delta_model2(k,i,j,words_de,words_en, summation):
	global q
	global t

	l = len(words_en)
	m = len(words_de)

	#calculate the numerator term
	t_val = 0;
	if j == -1:
		t_val = t[words_de[i], 'NULL']
	else:
		t_val = t[words_de[i], words_en[j]]
	numerator = q[j+1,i+1,l,m] * t_val

	return numerator / summation

'''This function is used to calculate initial values for the "t" parameters for IBM Model 1. It iterates over each sentence pair, keeping track of which foreign words 
can be aligned to which english word. The "t" t(f|e) parameters are initialized to a normal distribution over all foreign words that could be aligned to an english word "e"
in the corpus'''
def calculate_counts():
	global lines_en
	global lines_de

	word_map['NULL'] = defaultdict(str)
	word_cnt_map['NULL'] = 0
	i = 0

	with open("corpus.en") as en_file:
		lines_en = en_file.readlines()

	with open("corpus.de") as de_file:
		lines_de = de_file.readlines()	

	while i < len(lines_en):
		line_en = lines_en[i]
		if not line_en.strip('\n'):
			break
		
		line_de = lines_de[i]
		line_en = line_en.strip('\n')
		line_de = line_de.strip('\n')
		i += 1

		words_en = line_en.split(' ')
		words_de = line_de.split(' ')

		if (len(words_en), len(words_de)) not in lmpairs:
			lmpairs[len(words_en), len(words_de)] = len(words_en), len(words_de)

		for word_en in words_en:
			for word_de in words_de:
				if word_de not in word_map['NULL']:
					word_map['NULL'][word_de] = word_de
					word_cnt_map['NULL'] += 1
					t[word_de, 'NULL'] = 0

				if word_en not in word_map:
					word_cnt_map[word_en] = 1
					t[word_de, word_en] = 0
					word_list = defaultdict(str)
					word_list[word_de] = word_de
					word_map[word_en] = word_list
				else:
					if word_de not in word_map[word_en]:
						word_cnt_map[word_en] += 1
						t[word_de, word_en] = 0
						word_map[word_en][word_de] = word_de

	for en_word in word_map:
		for de_word in word_map[en_word]:
			t[de_word, en_word] = 1 / word_cnt_map[en_word]


def main(german_file_name, scrambled_file_name, unscrambled_file_name, alignment_model1_filename, alignment_model2_filename, foreign_words_filename):
	print('Start time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('Starting the EM algorithm for IBM Model 1 - Question 4 at %s' %(strftime("%Y-%m-%d %H:%M:%S", localtime())))
	calculate_counts()
	em_algorithm_model1()
	print('Finding foreign words for English words')
	find_foreign_words(foreign_words_filename)
	print('Output is saved at ./foreign_words_output')
	print('Finding alignments under IBM Model 1')
	find_alignments('1', alignment_model1_filename)
	print('Output is saved at ./alignments_model1')
	print('Starting the EM algorithm for IBM Model 2 - Question 5 at %s' %(strftime("%Y-%m-%d %H:%M:%S", localtime())))
	initialize_q_params()
	em_algorithm_model2()
	print('Finding alignments under IBM Model 2')
	find_alignments('2', alignment_model2_filename)
	print('Output is saved at ./alignments_model2')
	print('Finding translations - Question 6 at %s' %(strftime("%Y-%m-%d %H:%M:%S", localtime())))
	find_translations(german_file_name, scrambled_file_name, unscrambled_file_name)
	print('Output is saved at ./unscrambled.en')
	print('End time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))

