import sys

import numpy as np
from collections import Counter
from scipy.special import softmax


"""
Your name and file comment here:
    Chakrya Ros
"""


"""
Cite your sources here:
"""

def generate_tuples_from_file(file_path):
  """
  Implemented for you. 

  counts on file being formatted like:
  1 Comparison  O
  2 with  O
  3 alkaline  B
  4 phosphatases  I
  5 and O
  6 5 B
  7 - I
  8 nucleotidase  I
  9 . O

  1 Pharmacologic O
  2 aspects O
  3 of  O
  4 neonatal  O
  5 hyperbilirubinemia  O
  6 . O

  params:
    file_path - string location of the data
  return:
    a list of lists of tuples in the format [[(token, label), (token, label)...], ...]
  """
  current = []
  f = open(file_path, "r", encoding="utf8")
  examples = []
  for line in f:
    if len(line.strip()) == 0 and len(current) > 0:
      examples.append(current)
      current = []
    else:
      pieces = line.strip().split()
      current.append(tuple(pieces[1:]))
  if len(current) > 0:   # THESE LINES WERE MISSING
    examples.append(current)   # THESE LINES WERE MISSING
  f.close()
  return examples


def get_words_from_tuples(examples):
  """
  You may find this useful for testing on your development data.

  params:
    examples - a list of tuples in the format [(token, label), (token, label)...]
  return:
    a list of lists of tokens
  """
  return [[t[0] for t in example] for example in examples]


def decode(data, probability_table, pointer_table):
  """
  TODO: implement
  params: 
    data - a list of tokens
    probability_table - a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
    pointer_table - a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
  pass


def precision(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of precision at the entity level
  """

  #get the entities in gold label
  gold_labels_entities = findEntities_help(gold_labels)

  #get the entities in classify label
  classify_labels_entities = findEntities_help(classified_labels)

  #the length of classify label entities
  labeled_spans = len(classify_labels_entities)

  # number of true positive
  Truepostive = len(set.intersection(gold_labels_entities, classify_labels_entities))

  return float(Truepostive/labeled_spans)

  # O O B I O B I I O  B I I O O O O
  # O O B O O O B I O  B I I 0 B O O

def recall(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of recall at the entity level
  """
  #get the entities in gold label
  gold_labels_entities = findEntities_help(gold_labels)

  #get the entities in classify label
  classify_labels_entities = findEntities_help(classified_labels)

  #the length of gold label entities
  gold_spans = len(gold_labels_entities)

  # number of true positive
  Truepostive = len(set.intersection(gold_labels_entities, classify_labels_entities))

  return float(Truepostive/gold_spans)

def f1(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of f1 at the entity level
  """
  prec = precision(gold_labels, classified_labels)
  recal = recall(gold_labels, classified_labels)
  f1 = (2.0 * prec * recal) / (prec + recal)
  return f1

def pretty_print_table(data, list_of_dicts):
  """
  Pretty-prints probability and backpointer lists of dicts as nice tables.
  Truncates column header words after 10 characters.
  params:
    data - list of words to serve as column headers
    list_of_dicts - list of dicts with len(data) dicts and the same set of
      keys inside each dict
  return: None
  """
  # ensure that each dict has the same set of keys
  keys = None
  for d in list_of_dicts:
    if keys is None:
      keys = d.keys()
    else:
      if d.keys() != keys:
        print("Error! not all dicts have the same keys!")
        return
  header = "\t" + "\t".join(['{:11.10s}']*len(data))
  header = header.format(*data)
  rows = []
  for k in keys:
    r = k + "\t"
    for d in list_of_dicts:
      if type(d[k]) is float:
        r += '{:.9f}'.format(d[k]) + "\t"
      else:
        r += '{:10.9s}'.format(str(d[k])) + "\t"
    rows.append(r)
  print(header)
  for row in rows:
    print(row)

"""
Implement any other non-required functions here

"""
def findEntities_help(data):
  entites_span = set()
  entityStart = 0
  entityEnd = 0
  currentState = True
  count = 0

  for word, label in data:
    count = count + 1
    if currentState == True:
      if label == 'B':
        currentState = False
        entityStart = count
    elif currentState == False:
      if label == 'B':
        entityEnd = count - 1
        entites_span.add((entityStart, entityEnd))
        entityStart = count
      if label == 'O':
        entityEnd = count - 1
        entites_span.add((entityStart, entityEnd))
        currentState = True
  if currentState == True:
    entites_span.add((entityStart, entityEnd))

  return entites_span


"""
Implement the following class
"""
class NamedEntityRecognitionHMM:
  
  def __init__(self):
    # TODO: implment as needed
    self.pi = Counter()
    # self.pi_prob = Counter()
    self.transitions = {}    #this is A 
    self.emissions = {}     #this is B, emission
    self.states = set()     # Q
    self.vocab = set()     # O
    self.token_list = []

  def train(self, examples):
    """
    Trains this model based on the given input data
    params: examples - a list of lists of (token, label) tuples
    return: None
    """
    for sentence in examples:
      for i in range(len(sentence)):
        word = sentence[i][0]
        state = sentence[i][1]
        self.token_list.append((word, state))
        self.vocab.add(word)
        if i==0:
          self.pi[state] +=1

        if state not in self.emissions:
          self.emissions[state] = Counter()
        self.emissions[state][word] +=1


    for i in range(len(self.token_list)):
      word = self.token_list[i][0]
      state = self.token_list[i][1]
      if i > 0:

        if self.token_list[i-1][1] not in self.transitions:
          self.transitions[self.token_list[i-1][1]] = Counter()
        self.transitions[self.token_list[i-1][1]][state] +=1

        
    # print(self.token_list)
    # print("self.pi : ", self.pi)
    # print("self.transition ", self.transitions)
    # print("self.emissions ", self.emissions)
    # print(len(self.vocab))

    



    
   
  def generate_probabilities(self, data): 
    """
    params: data - a list of tokens
    return: two lists of dictionaries --
      - first a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
      - second a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
    """
    first_list_state_prob = []
    second_list_state_state = []
    prev_prob = 0 
    prev_state = None
    for i in range(len(data)):
      word = data[i]
      max_prob = 0
      max_state = None
      dict_prob_emssion = {}
      dict_state_state = {}
      count = 0
      for state in self.emissions.keys():
        # print(word, state)
        if word not in self.emissions[state]:
          emission = (0 + 1) / (sum(self.emissions[state].values()) + len(self.vocab))
          s = sum(self.emissions[state].values())
          # print("if emission = {} / {} + {}".format(1, s,len(self.vocab)))
          
        else:
          emission = (self.emissions[state][word] + 1) / (sum(self.emissions[state].values()) + len(self.vocab))
          e = self.emissions[state][word] + 1
          s = sum(self.emissions[state].values())
          # print("else emission = {} / {} + {}".format(e, s,len(self.vocab)))
        if i == 0:
          #pi * emssion probability
          pi_val = self.pi[state] / sum(self.pi.values())
          prob = emission * pi_val
          dict_prob_emssion[state] = prob

          dict_state_state[state] = prev_state
          
        else:
      
          print(state)
          # print(state, prev_state)
          transition = self.transitions[prev_state][state] / sum(self.transitions[prev_state].values()) 
          # t = self.transitions[prev_state][state] 
          # s = sum(self.transitions[prev_state].values()) 
          # print("transition = {} / {}".format(t, s))
          prob = prev_prob * transition * emission
          dict_prob_emssion[state] = prob
          dict_state_state[state] = prev_state
        if max_state is None or prob > max_prob:
          max_state = state
          max_prob = prob
        
      first_list_state_prob.append(dict_prob_emssion)
      second_list_state_state.append(dict_state_state)  
      prev_prob = max_prob
      prev_state = max_state
      
    print(first_list_state_prob)
    print(second_list_state_state)
    return first_list_state_prob, second_list_state_state
    


  def __str__(self):
    return "HMM"

"""
Implement the following class
"""
class NamedEntityRecognitionMEMM:
  def __init__(self):
    # implement as needed
    pass

  def train(self, examples):
    """
    Trains this model based on the given input data
    params: examples - a list of lists of (token, label) tuples
    return: None
    """
    pass

  def featurize(self, data):
    """
    CHOOSE YOUR OWN PARAMS FOR THIS FUNCTION
    convert a list of tokens to a list of (feature, value) tuples

    """
    pass

  def generate_probabilites(self, data):
    pass

  def __str__(self):
    return "MEMM"


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python hw5_ner.py training-file.txt testing-file.txt")
    sys.exit(1)

  training = sys.argv[1]
  testing = sys.argv[2]
  training_examples = generate_tuples_from_file(training)
 
  testing_examples = generate_tuples_from_file(testing)

  # instantiate each class, train it on the training data, and 
  # evaluate it on the testing data

  NER_HMM = NamedEntityRecognitionHMM()
  NER_HMM.train(training_examples)
  # data = [('Comparison', 'O'), ('in', 'O'), ('alkaline', 'B'), ('phosphatases','I'), ('in', 'O'), ('5', 'B'), ('-', 'I'),('nucleotidase', 'I'), ('.', 'O')]
  data = ['Comparison', 'in' , 'alkaline' , 'phosphatases', 'in', '5' , '-' , 'nucleotidase', '.']

  NER_HMM.generate_probabilities(data)
  


  # ensure that back pointers are correct
  # point_answers = [{'O': None, 'I': None, 'B': None}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, 
  #                 {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}, 
                      #  {'O': 'O', 'I': 'I', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}, 
                      #  {'O': 'I', 'I': 'I', 'B': 'O'}]  
                      #       
#self.transition  {'O': Counter({'O': 4, 'B': 3}), 'B': Counter({'I': 3}), 'I': Counter({'O': 3, 'I': 2})}
#self.emissions  {'O': Counter({'in': 3, '.': 2, 'Comparison': 1, 'the': 1, 'diagnosis': 1}), 
# 'B': Counter({'alkaline': 1, '5': 1, 'Serum': 1}), 
# 'I': Counter({'phosphatases': 1, '-': 1, 'nucleotidase': 1, 'gamma': 1, 'glutamyltransferase': 1})}


