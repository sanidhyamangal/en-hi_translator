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
path_to_file = os.path.dirname('./')+'hin.txt'

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