# import print, div and absolute to be compat with all the versions of py
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import tensorflow 
import tensorflow as tf

# import test split dataset 
from sklearn.model_selection import train_test_split

# import helper libs 
import unicodedata # to process unicode dataset 
import re # a regx lib for the text 
import numpy as np # for matrix maths 
import io # for input and output of the files 
import time # for managing time
import os # for os related ops

# a path to the file 
path_to_file = './hin.txt'

# function to convert unicode into the text
def unicode_to_ascii(s):
    # normalize the data using unicode data
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# function to preprocess the sentence 
def preprocess_sentence(w):

    # get ascii version of the text
    w = unicode_to_ascii(w.lower().strip())

    # adding a padding for the punctuation
    w = re.sub(r"([?.!,¿।])", r" \1", w)
    w = re.sub(r'[" "]+', " ", w)

    # striping sentence
    w = w.rstrip().strip()

    # add start and end in the sentence
    w = '<start> ' + w + ' <end>'

    return w

# function to create dataset
def create_dataset(path, num_examples=None):
    # extract lines from the data file
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    # get the word pairs after processing the sentences
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    # return zipped word pairs
    return zip(*word_pairs)

# function to get max length of a tensor
def max_length(tensor):
    return max(len(t) for t in tensor)

# tokenize the language
def tokenize(lang):

    # create a language tokenizer
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    # fit the text lang to this tokenizer
    lang_tokenizer.fit_on_texts(lang)

    # get the tensor sequences
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # pad the tensor sequences
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


# function to load the dataset 
def load_dataset(path, num_examples=None):

    # create dataset
    inp_lang, targ_lang =  create_dataset(path, num_examples)

    # tokenize input lang
    input_tensor, input_lang_tokenizer = tokenize(inp_lang)

    # tokenize target lang
    target_tensor, target_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor,  input_lang_tokenizer, target_lang_tokenizer


# limit the size of training to 3000
num_examples = 3000

# load dataset
input_tensor, target_tensor, input_lang, target_lang = load_dataset(path_to_file, num_examples)

# calculate max length of input and target tensor
max_input_tensor, max_target_tensor = max_length(input_tensor), max_length(target_tensor)

# split the dataset
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# function to convert the tensor into text
def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print("%d -------> %s" % (t, lang.index_word[t]))