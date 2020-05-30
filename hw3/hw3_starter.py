# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math

"""
Your name and file comment here:
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
def generate_tuples_from_file(training_file_path):
  pass


def precision(gold_labels, classified_labels):
  pass


def recall(gold_labels, classified_labels):
  pass

def f1(gold_labels, classified_labels):
  pass


"""
Implement any other non-required functions here
"""



"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:


  def __init__(self):
    # do whatever you need to do to set up your class here
    pass

  def train(self, examples):
    pass

  def score(self, data):
    pass

  def classify(self, data):
    pass

  def featurize(self, data):
    pass

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

  def __init__(self):
    pass

  def train(self, examples):
    pass

  def score(self, data):
    pass

  def classify(self, data):
    pass

  def featurize(self, data):
    pass

  def __str__(self):
    return "NAME FOR YOUR CLASSIFIER HERE"


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
    sys.exit(1)

  training = sys.argv[1]
  testing = sys.argv[2]

  sa = SentimentAnalysis()
  print(sa)
  # do the things that you need to with your base class
  

  improved = SentimentAnalysisImproved()
  print(improved)
  # do the things that you need to with your improved class









