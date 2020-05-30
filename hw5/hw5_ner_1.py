import sys

import numpy as np
from collections import Counter
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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
  max_prob = 0.0
  previous = None
  numState = len(pointer_table[0])
  NumWord = len(data)-1
  list_tuples =[]
  getState = None
  backpointer = None
  # print('pointer ', pointer)
  #get the max probability of the last column
  while NumWord >=0:
    if NumWord == (len(data) -1) :
      max_prob = 0
      for state in probability_table[NumWord].keys():
        if probability_table[NumWord][state] > max_prob:
          max_prob = probability_table[NumWord][state]
          getState = state
      # print(getState)
      backpointer = pointer_table[NumWord][getState]
      # print('backpointer ', backpointer)
      list_tuples.append((data[NumWord], getState))
    else:
      list_tuples.append((data[NumWord], backpointer))
      backpointer = pointer_table[NumWord][backpointer]

    NumWord -=1

  

  # print(list_tuples)
  list_tuples.reverse()
  return list_tuples
      


  


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

# helper function for find entities in the list
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

# helper function for find emission probability given word
def get_emission_prob(word,state, emission_dict, vocabLen):
  emission_prob = 0.0
  if word not in emission_dict[state]:
    emission = (0 + 1) / (sum(emission_dict[state].values()) + vocabLen)
  else:
    emission = (emission_dict[state][word] + 1) / (sum(emission_dict[state].values()) + vocabLen)
  return emission

#helper function for find transition probability given state, previous state
def get_transition_prob(currentState, preState, transition_dict, numPrevState):
  print('transition_dict[preState][currentState] ',transition_dict[currentState][preState])
  print('Num pre state ', numPrevState)
  print('transition ', transition_dict)
  return transition_dict[currentState][preState]/numPrevState

"""
Implement the following class
"""
class NamedEntityRecognitionHMM:
  
  def __init__(self):
    # TODO: implment as needed
    self.pi = Counter()
    self.pi_prob = {}
    self.transitions = {}    #this is A 
    self.emissions = {}     #this is B, emission
    self.states = set()     # Q
    self.vocab = set()     # O
    self.token_list = []
    self.state_count = {}




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
        self.states.add(state)
        if i==0:
          self.pi[state] +=1
        else:
          if sentence[i-1][1] not in self.transitions:
            self.transitions[sentence[i-1][1]] = Counter()
          self.transitions[sentence[i-1][1]][state] +=1

        if state not in self.emissions:
          self.emissions[state] = Counter()
        self.emissions[state][word] +=1


    for i in range(len(self.token_list)):
      word = self.token_list[i][0]
      state = self.token_list[i][1]
      # if i > 0:

        # if self.token_list[i-1][1] not in self.transitions:
        #   self.transitions[self.token_list[i-1][1]] = Counter()
        # self.transitions[self.token_list[i-1][1]][state] +=1

      if state not in self.state_count:
        self.state_count[state] = 1
      else:
        self.state_count[state] +=1

    # initial state pi probability
    for state in self.states:
      self.pi_prob[state] = self.pi[state] / sum(self.pi.values())

    # print("self.state_count ", self.state_count)
    # print("self.pi_prob: ", self.pi_prob)
    # print("self.transition ", self.transitions)
    # print("self.emissions ", self.emissions)
    # print("self.emissions prob ", self.emissions_prob)
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
    Numtoken = len(data)
    NumState = len(self.states)
    labels = sorted(list(self.states), reverse=True)
    prob_word_state = np.zeros((NumState, Numtoken), dtype=np.float64)
    prob_state_state =  np.zeros((NumState, Numtoken), dtype=np.int)
    NumVocab = len(self.vocab)
    list_state_prob = []
    list_state_state = []
    dict_prob_emssion = {}
    dict_state_state = {}
    for i in range(NumState):
      state = labels[i]
      emission_prob = get_emission_prob(data[0], state, self.emissions, NumVocab)
      prob_word_state[i, 0] =  self.pi_prob[state] * emission_prob
      dict_prob_emssion[state] = prob_word_state[i, 0]
      dict_state_state[state] = None
    list_state_prob.append(dict_prob_emssion)
    list_state_state.append(dict_state_state)

    # for j in range(1, Numtoken):
    #   word = data[j]
    #   dict_prob_emssion = {}
    #   dict_state_state = {}
    #   for i in range(NumState):
    #       emission_prob = get_emission_prob(word, labels[i], self.emissions, NumVocab)
    #       prevState = labels[0]
    #       state = labels[i]
    #       print('prev state: ', prevState, ' state ', state)
    #       transition_prob = get_transition_prob(state, prevState, self.transitions, self.state_count[state])
    #       print('transition_prob ', transition_prob)
    #       max_prob, max_index = prob_word_state[0, j-1] * transition_prob * emission_prob, 0
    #       for k in range(1,NumState):
            
    #         prevState = labels[i]
    #         state = labels[k]
    #         print('prev state in k: ', prevState, ' state  in k', state)
    #         transition_prob = get_transition_prob(state, prevState, self.transitions, self.state_count[state])
    #         print('transition_prob in k', transition_prob)
    #         prob = prob_word_state[k, j-1] * transition_prob * emission_prob
    #         print('prob in k', prob)
    #         if prob > max_prob:
    #           max_prob = prob
    #           max_index = k
       
    #       prob_word_state[i, j] = max_prob
    #       prob_state_state[i, j] = max_index
    #       dict_state_state[labels[i]] = labels[prob_state_state[i,j]]
    #       dict_prob_emssion[labels[i]] = prob_word_state[i, j]

    #   list_state_prob.append(dict_prob_emssion)
    #   list_state_state.append(dict_state_state)
          
    #fill the rest of probility into tables
    for j in range(1, Numtoken):
      word = data[j]
      dict_prob_emssion = {}
      dict_state_state = {}
      for i in range(NumState):
        emission_prob = get_emission_prob(word, labels[i], self.emissions, NumVocab)
        for k in range(NumState):
          prevState = labels[i]
          state = labels[k]
          print('prev state: ', prevState, ' state ', state)
          transition_prob = get_transition_prob(state, prevState, self.transitions, self.state_count[state])
          print('transition_prob ', transition_prob)
          prob = prob_word_state[k, j-1] * transition_prob * emission_prob
          print('prob in k', prob)
          if prob > prob_word_state[i, j]:
            prob_word_state[i, j] = prob
            prob_state_state[i,j] = k
          
        dict_state_state[labels[i]] = labels[prob_state_state[i,j]]
        dict_prob_emssion[labels[i]] = prob_word_state[i, j]
      list_state_prob.append(dict_prob_emssion)
      list_state_state.append(dict_state_state)
    # print(prob_state_state)
    # print(prob_word_state)
    print(list_state_prob)
    print(list_state_state)
    return list_state_prob, list_state_state

   


  def __str__(self):
    return "HMM"

"""
Implement the following class
"""
class NamedEntityRecognitionMEMM:
  def __init__(self):
    # implement as needed
    self.state = set()
    self.num_feats = []
    self.weights = None
    self.vocal = set()


    
  def train(self, examples):
    """
    Trains this model based on the given input data
    params: examples - a list of lists of (token, label) tuples
    return: None
    """
    # get state and word and store into set
    for sentence in examples:
      for i in range(len(sentence)):
        word = sentence[i][0]
        state = sentence[i][1]
        self.state.add(state)
        self.vocal.add(word)
    #initialize random weight
    vocab_len = len(self.vocal)
    print(vocab_len)
    self.weights = 2 * np.random.random_sample((len(self.state), (vocab_len + 1))) - 1


   
    # iterations = 100
    # index = 0
    # while index < 100:
    #   for i in range(len(examples)):

    #     pass

  def featurize(self, data):
    """
    CHOOSE YOUR OWN PARAMS FOR THIS FUNCTION
    convert a list of tokens to a list of (feature, value) tuples

    """
    count_vect = CountVectorizer()
    feature_counts = count_vect.fit_transform(data)
    feature_counts_nd = feature_counts.toarray()
    
    list_tuples = []
    for i in range(len(data)):
      list_tuples.append((data[i], feature_counts_nd[i]))
    print(list_tuples)
    return list_tuples

  def generate_probabilities(self, data):
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
  data = ['Comparison', 'in' , 'alkaline' , 'phosphatases', 'in', '5' , '-' , 'nucleotidase', '.']
  # vocab = len(NER_HMM.vocab)
  emission_dict = NER_HMM.emissions
  # print(get_emission_prob('in','O', emission_dict, vocab))
  prob_dict, pointer = NER_HMM.generate_probabilities(data)
  # print(prob_dict)
  # print(pointer)
  decode(data, prob_dict, pointer)


  NER_MEMM = NamedEntityRecognitionMEMM()
  # NER_MEMM.train(training_examples)
  # NER_MEMM.featurize(data)
  


  # ensure that back pointers are correct
  # point_answers = [{'O': None, 'I': None, 'B': None}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, 
  #                 {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}, 
                      #  {'O': 'O', 'I': 'I', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}, 
                      #  {'O': 'I', 'I': 'I', 'B': 'O'}]  
                      #       



