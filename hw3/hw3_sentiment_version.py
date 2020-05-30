# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
import re
import nltk 


"""
Your name and file comment here:
Name : Chakrya Ros
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
def generate_tuples_from_file(training_file_path):
    file = open(training_file_path, 'r')
    
    sentences = file.read()
    file.close()
    list_word = []
    tuple_word = ()
    wordsList = sentences.split("\n")
   
    for words in wordsList:
        word = words.split('\t')
        if len(word) == 3:
            tupleWord = (word[0],word[1],word[2])
            list_word.append(tupleWord)
#     print(list_word[0][1])
    return list_word
        

def precision(gold_labels, classified_labels):
    truePostive = 0
    falsePostive = 0
    for i in range(len(gold_labels)):
        if gold_labels[i] == str(1) and classified_labels[i] == str(1):
            truePostive +=1
        elif gold_labels[i] == str(0) and classified_labels[i] == str(1):
            falsePostive +=1
            print("falsePostive,",falsePostive)
    return (truePostive/(truePostive+falsePostive))


def recall(gold_labels, classified_labels):
    truePostive = 0
    falseNegative = 0
    for i in range(len(gold_labels)):
        if gold_labels[i] == '1' and classified_labels[i] == '1':
            truePostive +=1
        elif gold_labels[i] == '1' and classified_labels[i] == '0':
            falseNegative +=1
    return (truePostive/(truePostive+falseNegative))


def f1(gold_labels, classified_labels):
    prec = precision(gold_labels, classified_labels)
    recal = recall(gold_labels, classified_labels)
    f1 = 2 * ((prec*recal)/(prec+recal))
    return f1

"""
Implement any other non-required functions here
"""



"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:


    def __init__(self):
    # do whatever you need to do to set up your class here
        self.tupleWordPostive = []  
        self.tupleWordNegative = []
        self.probPositive = 0.0
        self.probNegative = 0.0
        self.NumDocu = 0          #number document
        self.NumPositive = 0      #number one class
        self.NumNegative = 0      # number zero class
        self.freqPos = {}         # frequency words in postive class
        self.freqNeg = {}         # frequency words in negative class
        self.tokenPostive = 0     #number words in postive class
        self.tokenNegative = 0    #numver words in negative class
        self.vocab = 0.0          # length of vocabulary
        self.vocab_list = []      #list of word in all document
        self.prior = {}           #calculate the probability of class
        self.likelihood = {}      #calculate the probability of word given class
        

    def train(self, examples):
        for i in range(len(examples)):
            if examples[i][2] == '1':
                self.NumPositive += 1 #count positive class
                self.tupleWordPostive.append(examples[i][1])
            else:
                self.NumNegative += 1 #count negative class
                self.tupleWordNegative.append(examples[i][1])

        
        #get the words and frequency word and store in frequency positive dictionary
        for sentence in self.tupleWordPostive:
            words = sentence.split()
            for w in words:
                if w in self.freqPos:
                    self.freqPos[w] +=1
                else:
                    self.freqPos[w] = 1
            for w in words:
                if w in self.vocab_list:
                    continue
                else:
                    self.vocab_list.append(w)
                
        #get the words and frequency word and store in frequency negative dictionary
        for sentence in self.tupleWordNegative:
            words = sentence.split()
            for w in words:
                if w in self.freqNeg:
                    self.freqNeg[w] +=1
                else:
                    self.freqNeg[w] = 1
            for w in words:
                if w in self.vocab_list:
                    continue
                else:
                    self.vocab_list.append(w)
                    
        #get tokens word from frequecy postive 
        self.tokenPostive = sum(self.freqPos.values())
        
        #get tokens word from frequecy Negative 
        self.tokenNegative = sum(self.freqNeg.values())
        
        #all vocabulatary in both class
        self.vocab = len(self.vocab_list)
        
        #number of document
        self.NumDocu = len(examples)
        
        #calcualte the probability of positive class
        self.probPositive = self.NumPositive/self.NumDocu
        
        
        #calcualte the probability of positive class
        self.probNegative = self.NumNegative/self.NumDocu
        
        #calculate the prior of each class and add into dictionary
        self.prior['1'] = self.probPositive
        self.prior['0'] = self.probNegative
        
        #calcualte the likelihood of each class and add into dictionatry
        for word in self.vocab_list:
            if word in self.freqPos:
                likelihood_one = (self.freqPos[word] + 1) / (self.tokenPostive + self.vocab)
                self.likelihood[(word,'1')] = likelihood_one
            else: 
                likelihood_one = ( 0 + 1) / (self.tokenPostive + self.vocab)
                self.likelihood[(word,'1')] = likelihood_one

            
            if word in self.freqNeg:
                likelihood_zero = (self.freqNeg[word] + 1) / (self.tokenNegative + self.vocab)
                self.likelihood[(word,'0')] = likelihood_zero
            else:
                likelihood_zero = (0 + 1) / (self.tokenNegative + self.vocab)
                self.likelihood[(word,'0')] = likelihood_zero
        
        
        
#         print(self.prior['1'])
#         print(self.likelihood)
#         print("Number of Positve: {}".format(self.NumPositive))
#         print("Number of Negative: {}".format(self.NumPositive))
#         print("Number of document: {}".format(self.NumDocu))
#         print("Probability of Positve : {}".format(self.probPositive))
#         print("Probability of negative : {}".format(self.probNegative))
        
#         print(self.freqPos)
#         print(self.tokenPostive)
#         print(self.tokenNegative)
#         print(self.vocab)
#         print("vocabulary list: {}".format(self.vocab_list))

    
    # calculate the score from the data
    #return the probability of each class
    def score(self, data):
        words = data.split()
        p_data_one = 0.0
        p_data_zero = 0.0
        likelihood_one = 0
        likelihood_zero = 0
        for w in words:
            if w in self.vocab_list:
                likelihood_one += np.log(self.likelihood[(w,'1')])
                likelihood_zero += np.log(self.likelihood[(w,'0')])
            else:
                continue
                
        p_data_one = np.exp(np.log(self.prior['1']) + likelihood_one)
        p_data_zero = np.exp(np.log(self.prior['0']) + likelihood_zero)
        
        return { '1' : p_data_one, '0' : p_data_zero}
                
                
    def classify(self, data):
        dict_class = self.score(data)
        if dict_class['1'] > dict_class['0']:
            return '1'
        else:
            return '0'
        
    # store feature and its value into list of tuple
    def featurize(self, data):
        words = data.split();
        featureList = {}
        tupleWordList = []
        for w in words:
            if w not in featureList:
                featureList[w] = 1
            else:
                featureList[w] +=1
        for w in featureList:
            tupleWordList.append((w, featureList[w]))
        return tupleWordList
        

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

    def __init__(self):
        
        self.probPositive = 0.0
        self.probNegative = 0.0
        self.NumDocu = 0          #number document
        self.NumPositive = 0      #number one class
        self.NumNegative = 0      # number zero class
        self.freqPos = {}         # frequency words in postive class
        self.freqNeg = {}         # frequency words in negative class
        self.tokenPostive = 0     #number words in postive class
        self.tokenNegative = 0    #numver words in negative class
        self.vocab = 0.0          # length of vocabulary
        self.vocab_list = []      #list of word in all document
        self.prior = {}           #calculate the probability of class
        self.likelihood = {}      #calculate the probability of word given class
        
    #helper function to clean the data   
    def preprocessData(self, examples):
        
        clean_data = []
        for i in range(len(examples)):
            #convert lower case
            sentence = examples[i][1].lower()
           
            #remove punctuation
            sentence = re.sub('[^a-zA-Z0-9\']', ' ', sentence)
            sentence = re.sub(r'\s+',' ', sentence)
            clean_data.append((examples[i][0], sentence, examples[i][2]))
#         print(clean_data)
        return clean_data
            
           
            
             
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
    exampleText = generate_tuples_from_file("train_file.txt")
    sa.train(exampleText)
    sa.featurize("I loved it loved I")

    
 
    
    improved = SentimentAnalysisImproved()
    print(improved)
    # do the things that you need to with your improved class
    exampleText = generate_tuples_from_file("train_file.txt")
    improved.preprocessData(exampleText)
    
    







