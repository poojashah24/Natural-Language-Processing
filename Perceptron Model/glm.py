from __future__ import division
from time import localtime, strftime
__author__="Pooja Shah <pbs2124@columbia.edu> "

""" Module imports """
import pdb
import math
import re
import os
from collections import defaultdict
from collections import OrderedDict
from subprocess import PIPE
import sys, subprocess

''' dictionaries for part 4 '''
tag_dict = defaultdict(float)
bigram_dict = defaultdict(float)

''' dictionaries for part 5 '''
bigram_dict_q5 = defaultdict(float)
tag_dict_q5 = defaultdict(float)
suffix_dict = defaultdict(float)

''' dictionaries for part 6 '''
prefix_dict = defaultdict(float)
num_dict = defaultdict(float)
hyphen_dict = defaultdict(float)

temp = open('temp','w')

def process(args):
	"Create a 'server' to send commands to."
	return subprocess.Popen(args, stdin=PIPE, stdout=PIPE)

def call(process, stdin):
	"Send command to a server and get stdout."
	output = process.stdin.write(stdin + "\n\n")
	line = ""
	while 1: 
		l = process.stdout.readline()
		if not l.strip(): break
		line += l
	return line

history_server = process(["python", "tagger_history_generator.py",  "ENUM"])
gold_server = process(["python", "tagger_history_generator.py",  "GOLD"])
decoder_server = process(["python", "tagger_decoder.py",  "HISTORY"])

''' utility functions used in features '''
def hasHyphen(word):
	if word.find('-') > -1:
		return True
	return False

def hasNumbers(word):
	return any(char.isdigit() for char in word)

def getWordSuffix(word, j):
	return word[-j:]

def getWordPrefix(word, j):
	return word[:j]

''' this is a utility function to check if two histories are equal '''
def equals(gold_history, tagging):
	global temp
	temp.write(gold_history)
	temp.write(tagging)
	temp.write('\n')
	history_lines = gold_history.split('\n')
	tag_lines = tagging.split('\n')
	for i in xrange(0,len(history_lines)):
		words_hist = history_lines[i].split()
		words_tag = tag_lines[i].split()
		for j in xrange(len(words_hist)):
			if words_hist[j] != words_tag[j]:
				return False
	return True


''' this method creates features for q5 and q6 '''
def create_features():
	train_data = open('tag_train.dat', 'r')
	for line in train_data:
		line = line.strip('\n')
		if line:
				words = line.split()
				w = words[0]
				tag = words[1]
				if len(w) >= 1:
					suffix_dict[getWordSuffix(w,1)+":1:"+tag] = 0.0
					prefix_dict[getWordPrefix(w,1)+":1:"+tag] = 0
				if len(w) >= 2:
					suffix_dict[getWordSuffix(w,2)+":2:"+tag] = 0.0
					prefix_dict[getWordPrefix(w,2)+":2:"+tag] = 0
				if len(w) >= 3:
					suffix_dict[getWordSuffix(w,3)+":3:"+tag] = 0.0
					prefix_dict[getWordPrefix(w,3)+":3:"+tag] = 0.0
				if hasNumbers(w):
					num_dict[tag] = 0.0
				if hasHyphen(w):
					hyphen_dict[tag] = 0.0


	train_data.seek(0)
	prev = '*'
	for line in train_data:
		line = line.strip('\n')
		line = line.strip()
		if line:
			words = line.split()
			w = words[0]
			tag = words[1]
			tag_dict_q5[w+":"+tag] = 0.0
			bigram_dict_q5[prev+":"+tag] = 0.0
			prev = tag
		else:
			prev = '*'
	train_data.close()


''' this method creates the tag and bigram features from the pre trained model in q4 '''
def generate_tag_and_bigram_dict(model_file):
	global tag_dict, bigram_dict
	model = open(model_file, 'r')
	for line in model:
		line = line.strip('\n')
		if line.find('TAG') > -1:
			words = line.split(' ')
			tag_dict[words[0][4:]] = float(words[1])
		elif line.find('BIGRAM') > -1:
			words = line.split(' ')
			bigram_dict[words[0][7:]] = float(words[1])

''' this method is used to get the highest scoring tagging for each sentence in the development data. The model used for this tagging is the pre trained model provided '''
def get_highest_scoring_tagging(dev_file,res_file):
	global tag_dict, bigram_dict, history_server, decoder_server, process, call
	dev_data = open(dev_file,'r')
	dev_res = open(res_file, 'w')
	sentence = ''
	for line in dev_data:
		if line.strip('\n'):
			sentence += line
		else:
			sentence = sentence.rstrip('\n')
			sentence_history = call(history_server, sentence);
			lines = sentence_history.split('\n')
			words = sentence.split('\n')
			hist = ''
			for l in lines:
				if l.strip():
					tokens = l.split()
					i = tokens[0]
					ti_1 = tokens[1]
					ti = tokens[2]

					word = words[int(i)-1]
					score = 0.0

					''' score the history based on bigram and tag features '''
					if bigram_dict[ti_1+":"+ti] != 0:
						score = bigram_dict[ti_1+":"+ti]

					if tag_dict[word+":"+ti] != 0:
						score += tag_dict[word+":"+ti]
					hist += l + " " + str(score) + '\n'

			hist = hist.rstrip('\n')
			res = ''
			tagging = call(decoder_server, hist)
			tag_lines = tagging.split('\n')
			for tag_line in tag_lines:
				if tag_line.strip():
					tag_words = tag_line.split()
					if (int(tag_words[0]) <= len(words)):
						w = words[int(tag_words[0])-1]
						tag = tag_words[2]
						res += w + '\t' + tag + '\n'

			dev_res.write(res)
			dev_res.write('\n')
			sentence = ''


''' this function is used to train the model for q5, based on tag features, bigram features, and suffix features '''
def train_model(train_file, model_file):
	global suffix_dict, bigram_dict_q5, tag_dict_q5
	sentence = ''
	wordStr = ''
	train_data = open(train_file, 'r')

	for step in xrange(5):
		train_data.seek(0)
		for line in train_data:
			if line.strip('\n'):
				tok = line.strip('\n').split('\t')
				wordStr = wordStr + tok[0] + ' '
				sentence += line
			else:
				words = wordStr.split()
				sentence = sentence.rstrip('\n')
				gold_history = call(gold_server, sentence);
				sentence_history = call(history_server, sentence);
				lines = sentence_history.split('\n')
				hist = ''
				for l in lines:
					if l.strip():
						tokens = l.split()
						i = tokens[0]
						ti_1 = tokens[1]
						ti = tokens[2]

						word = words[int(i)-1]
						score = 0.0

						key = ti_1+":"+ti
						score = bigram_dict_q5[key]

						key = word+":"+ti
						score += tag_dict_q5[key]

						if len(word) >= 1:
							key = getWordSuffix(word,1)+":1:"+ti
							score += suffix_dict[key]

						if len(word) >= 2:
							key = getWordSuffix(word,2)+":2:"+ti
							score += suffix_dict[key]

						if len(word) >=3:
							key = getWordSuffix(word,3)+":3:"+ti
							score += suffix_dict[key]

						hist += l + " " + str(score) + '\n'

				hist = hist.rstrip('\n')
				res = ''
				tagging = call(decoder_server, hist)
				if not equals(gold_history,tagging):
					lines = gold_history.split('\n')
					for line in lines:
						line = line.strip('\n')
						if line:
							tokens = line.split()
							i = tokens[0]
							ti_1 = tokens[1]
							ti = tokens[2]
							word = words[int(i)-1]

							key = ti_1+":"+ti
							if key in bigram_dict_q5:
								bigram_dict_q5[key] += 1

							key = word+":"+ti
							if key in tag_dict_q5:
								tag_dict_q5[key] += 1

							if len(word) >= 1:
								key = getWordSuffix(word,1)+":1:"+ti
								if key in suffix_dict:
									suffix_dict[key] += 1

							if len(word) >= 2:
								key = getWordSuffix(word,2)+":2:"+ti
								if key in suffix_dict:
									suffix_dict[key] += 1

							if len(word) >= 3:
								key = getWordSuffix(word,3)+":3:"+ti
								if key in suffix_dict:
									suffix_dict[key] += 1

					tag_lines = tagging.split('\n')
					for line in tag_lines:
						if not "STOP" in line:
							line = line.strip('\n')
							if line:
								tokens = line.split()
								i = tokens[0]
								ti_1 = tokens[1]
								ti = tokens[2]
								word = words[int(i)-1]

								key = ti_1+":"+ti
								if key in bigram_dict_q5:
									bigram_dict_q5[ti_1+":"+ti] -= 1

								key = word+":"+ti
								if key in tag_dict_q5:
									tag_dict_q5[word+":"+ti] -= 1
									
								if len(word) >= 1:
									key = getWordSuffix(word,1)+":1:"+ti
									if key in suffix_dict:
										suffix_dict[key] -= 1

								if len(word) >= 2:
									key = getWordSuffix(word,2)+":2:"+ti
									if key in suffix_dict:
										suffix_dict[key] -= 1

								if len(word) >= 3:
									key = getWordSuffix(word,3)+":3:"+ti
									if key in suffix_dict:
										suffix_dict[key] -= 1

				sentence = ''
				wordStr = ''
	train_data.close();

	model_file = open(model_file,'w')
	for x in bigram_dict_q5:
		model_file.write('BIGRAM:'+x+' '+str(bigram_dict_q5[x]))
		model_file.write('\n')

	for x in tag_dict_q5:
		model_file.write('TAG:'+x+' '+str(tag_dict_q5[x]))
		model_file.write('\n')

	for x in suffix_dict:
		model_file.write('SUFF:'+x+' '+str(suffix_dict[x]))
		model_file.write('\n')
	model_file.close()	

''' this function gets the highest scoring tagging for the development data based on a model trained on tag, bigram, and suffix features '''
def get_highest_scoring_tagging_q5(dev_file, res_file):
	global tag_dict_q5, bigram_dict_q5, suffix_dict, history_server, decoder_server, process, call
	dev_data = open(dev_file,'r')
	dev_res = open(res_file, 'w')
	sentence = ''
	for line in dev_data:
		if line.strip('\n'):
			sentence += line
		else:
			sentence = sentence.rstrip('\n')
			sentence_history = call(history_server, sentence);
			lines = sentence_history.split('\n')
			words = sentence.split('\n')
			hist = ''
			for l in lines:
				if l.strip():
					tokens = l.split()
					i = tokens[0]
					ti_1 = tokens[1]
					ti = tokens[2]

					word = words[int(i)-1]
					score = 0.0

					key = ti_1+":"+ti
					score += bigram_dict_q5[key]

					key = word+":"+ti
					score += tag_dict_q5[key]
					
					if len(word) >= 1:
						key = getWordSuffix(word,1)+":1:"+ti
						score += suffix_dict[key]

					if len(word) >= 2:
						key = getWordSuffix(word,2)+":2:"+ti
						score += suffix_dict[key]

					if len(word) >= 3:
						key = getWordSuffix(word,3)+":3:"+ti
						score += suffix_dict[key]
					hist += l + " " + str(score) + '\n'

			hist = hist.rstrip('\n')
			res = ''
			tagging = call(decoder_server, hist)
			tag_lines = tagging.split('\n')
			for tag_line in tag_lines:
				if tag_line.strip():
					tag_words = tag_line.split()
					if (int(tag_words[0]) <= len(words)):
						w = words[int(tag_words[0])-1]
						tag = tag_words[2]
						res += w + '\t' + tag + '\n'''

			dev_res.write(res)
			dev_res.write('\n')
			sentence = ''

''' this function is used to train the model for q5, based on tag features, bigram features, suffix and prefix features. Optionally, a feature that looks for numbers
in words, and a feature that looks for hyphens in words is also used '''
def train_model_q6(train_file, model_file_name, mode):
	global bigram_dict_q5, tag_dict_q5, num_dict, suffix_dict, prefix_dict
	sentence = ''
	wordStr = ''
	train_data = open('tag_train.dat','r')

	for step in xrange(5):
		train_data.seek(0)
		for line in train_data:
			if line.strip('\n'):
				tok = line.strip('\n').split('\t')
				wordStr = wordStr + tok[0] + ' '
				sentence += line
			else:
				words = wordStr.split()
				sentence = sentence.rstrip('\n')
				gold_history = call(gold_server, sentence);
				sentence_history = call(history_server, sentence);
				lines = sentence_history.split('\n')
				hist = ''
				for l in lines:
					if l.strip():
						tokens = l.split()
						i = tokens[0]
						ti_1 = tokens[1]
						ti = tokens[2]

						word = words[int(i)-1]
						score = 0.0

						key = ti_1+":"+ti
						score = bigram_dict_q5[key]

						key = word+":"+ti
						score += tag_dict_q5[key]

						if len(word) >= 1:
							key = getWordSuffix(word,1)+":1:"+ti
							score += suffix_dict[key]
							key = getWordPrefix(word,1)+":1:"+ti
							score += prefix_dict[key]

						if len(word) >= 2:
							key = getWordSuffix(word,2)+":2:"+ti
							score += suffix_dict[key]
							key = getWordPrefix(word,2)+":2:"+ti
							score += prefix_dict[key]

						if len(word) >= 3:
							key = getWordSuffix(word,3)+":3:"+ti
							score += suffix_dict[key]
							key = getWordPrefix(word,3)+":3:"+ti
							score += prefix_dict[key]

						if int(mode) == 2:
							if hasNumbers(word):
								score += num_dict[ti]

						if int(mode) == 3:
							if hasHyphen(word):
								score += hyphen_dict[ti]

						hist += l + " " + str(score) + '\n'

				hist = hist.rstrip('\n')
				res = ''
				tagging = call(decoder_server, hist)

				if equals(gold_history,tagging):
					sentence = ''
					wordStr = ''
					continue
				else:
					lines = gold_history.split('\n')
					for line in lines:
						line = line.strip('\n')
						if line:
							tokens = line.split()
							i = tokens[0]
							ti_1 = tokens[1]
							ti = tokens[2]
							word = words[int(i)-1]

							key = ti_1+":"+ti
							if key in bigram_dict_q5:
								bigram_dict_q5[key] += 1

							key = word+":"+ti
							if key in tag_dict_q5:
								tag_dict_q5[key] += 1

							if len(word) >= 1:
								key = getWordSuffix(word,1)+":1:"+ti
								if key in suffix_dict:
									suffix_dict[key] += 1
								key = getWordPrefix(word,1)+":1:"+ti
								if key in prefix_dict:
									prefix_dict[key] += 1

							if len(word) >= 2:
								key = getWordSuffix(word,2)+":2:"+ti
								if key in suffix_dict:
									suffix_dict[key] += 1
								key = getWordPrefix(word,2)+":2:"+ti
								if key in prefix_dict:
									prefix_dict[key] += 1

							if len(word) >= 3:
								key = getWordSuffix(word,3)+":3:"+ti
								if key in suffix_dict:
									suffix_dict[key] += 1
								key = getWordPrefix(word,3)+":3:"+ti
								if key in prefix_dict:
									prefix_dict[key] += 1

							if int(mode) == 2:
								if hasNumbers(word) and ti in num_dict:
									num_dict[ti] += 1

							if int(mode) == 3:
								if hasHyphen(word) and ti in hyphen_dict:
									hyphen_dict[ti] += 1	

					tag_lines = tagging.split('\n')
					for line in tag_lines:
						if not "STOP" in line:
							line = line.strip('\n')
							if line:
								tokens = line.split()
								i = tokens[0]
								ti_1 = tokens[1]
								ti = tokens[2]
								word = words[int(i)-1]

								key = ti_1+":"+ti
								if key in bigram_dict_q5:
									bigram_dict_q5[key] -= 1

								key = word+":"+ti
								if word+":"+ti in tag_dict_q5:
									tag_dict_q5[key] -= 1

								if len(word) >= 1:
									key = getWordSuffix(word,1)+":1:"+ti
									if key in suffix_dict:
										suffix_dict[key] -= 1
									key = getWordPrefix(word,1)+":1:"+ti
									if key in prefix_dict:
										prefix_dict[key] -= 1

								if len(word) >= 2:
									key = getWordSuffix(word,2)+":2:"+ti
									if key in suffix_dict:
										suffix_dict[key] -= 1
									key = getWordPrefix(word,2)+":2:"+ti
									if key in prefix_dict:
										prefix_dict[key] -= 1

								if len(word) >= 3:
									key = getWordSuffix(word,3)+":3:"+ti
									if key in suffix_dict:
										suffix_dict[key] -= 1
									key = getWordPrefix(word,3)+":3:"+ti
									if key in prefix_dict:
										prefix_dict[key] -= 1

								if int(mode) == 2:
									if hasNumbers(word) and ti in num_dict:
										num_dict[ti] -= 1

								if int(mode) == 3:
									if hasHyphen(word) and ti in hyphen_dict:
										hyphen_dict[ti] -= 1
								
				sentence = ''
				wordStr = ''
	train_data.close();

	model_file = open('prefix_suffix_tagger.model','w')
	for x in bigram_dict_q5:
		model_file.write('BIGRAM:'+x+' '+str(bigram_dict_q5[x]))
		model_file.write('\n')

	for x in tag_dict_q5:
		model_file.write('TAG:'+x+' '+str(tag_dict_q5[x]))
		model_file.write('\n')

	for x in suffix_dict:
		model_file.write('SUFF:'+x+' '+str(suffix_dict[x]))
		model_file.write('\n')
	model_file.close()

''' this function gets the highest scoring tagging for the development data based on a model trained on tag, bigram, suffix and prefix features (and optionally, a feature 
	that looks for hasNumbers in words, and a feature that looks for hyphens in words is also used) '''
def get_highest_scoring_tagging_q6(dev_file, res_file, mode):
	global tag_dict_q5, bigram_dict_q5, suffix_dict, prefix_dict, history_server, decoder_server, process, call
	global num_dict
	dev_data = open(dev_file,'r')
	dev_res = open(res_file, 'w')
	sentence = ''
	for line in dev_data:
		if line.strip('\n'):
			sentence += line
		else:
			sentence = sentence.rstrip('\n')
			sentence_history = call(history_server, sentence);
			lines = sentence_history.split('\n')
			words = sentence.split('\n')
			hist = ''
			for l in lines:
				if l.strip():
					tokens = l.split()
					i = tokens[0]
					ti_1 = tokens[1]
					ti = tokens[2]

					word = words[int(i)-1]
					score = 0.0

					key = ti_1+":"+ti
					score += bigram_dict_q5[key]

					key = word+":"+ti
					score += tag_dict_q5[key]

					if len(word) >= 1:
						key = getWordSuffix(word,1)+":1:"+ti
						score += suffix_dict[key]
						key = getWordPrefix(word,1)+":1:"+ti
						score += prefix_dict[key]

					if len(word) >= 2:
						key = getWordSuffix(word,2)+":2:"+ti
						score += suffix_dict[key]
						key = getWordPrefix(word,2)+":2:"+ti
						score += prefix_dict[key]

					if len(word) >= 3:
						key = getWordSuffix(word,3)+":3:"+ti
						score += suffix_dict[key]
						key = getWordPrefix(word,3)+":3:"+ti
						score += prefix_dict[key]

					if int(mode) == 2:
						if hasNumbers(word):
							score += num_dict[ti]

					if int(mode) == 3:
						if hasHyphen(word):
							score += hyphen_dict[ti]

					hist += l + " " + str(score) + '\n'

			hist = hist.rstrip('\n')
			res = ''
			tagging = call(decoder_server, hist)
			tag_lines = tagging.split('\n')
			for tag_line in tag_lines:
				if tag_line.strip():
					tag_words = tag_line.split()
					if (int(tag_words[0]) <= len(words)):
						w = words[int(tag_words[0])-1]
						tag = tag_words[2]
						res += w + '\t' + tag + '\n'

			dev_res.write(res)
			dev_res.write('\n')
			sentence = ''

''' this is the entry point for executing q4 '''
def main_q4(model_file, dev_file, res_file):
	print('Start time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))
	generate_tag_and_bigram_dict(model_file)
	get_highest_scoring_tagging(dev_file, res_file)
	print('End time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))

''' this is the entry point for executing q5 '''
def main_q5(train_file, model_file, dev_file, res_file):
	print('Start time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))
	create_features()
	train_model(train_file, model_file)
	get_highest_scoring_tagging_q5(dev_file, res_file)
	print('End time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))

''' this is the entry point for executing q6 '''
def main_q6_1(train_file, model_file, dev_file, res_file, mode):
	print('Start time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))
	create_features()
	train_model_q6(train_file, model_file, mode)
	get_highest_scoring_tagging_q6(dev_file, res_file, mode)
	print('End time:%s' %strftime("%Y-%m-%d %H:%M:%S", localtime()))