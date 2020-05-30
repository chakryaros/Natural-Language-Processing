# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
import re
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('stopwords')
nltk.download('wordnet')


"""
Your name and file comment here:
Name : Chakrya Ros
"""


"""
Cite your sources here:
 - https://gist.github.com/sebleier/554280
 - https://github.com/llSourcell/logistic_regression/blob/master/Sentiment%20analysis%20with%20Logistic%20Regression.ipynb
 - https://github.com/abdulfatir/twitter-sentiment-analysis/tree/master/dataset
"""


"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""

# read the data and split the into three tuple and add into the list
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

    return list_word
        
# Calculate the precision from given gold labels and classifed labels
def precision(gold_labels, classified_labels):
    truePostive = 0
    falsePostive = 0
    for i in range(len(gold_labels)):
        if gold_labels[i] == str(1) and classified_labels[i] == str(1):
            truePostive +=1
        elif gold_labels[i] == str(0) and classified_labels[i] == str(1):
            falsePostive +=1
    return (truePostive/(truePostive+falsePostive))

# Calculate the recall from given gold labels and classifed labels
def recall(gold_labels, classified_labels):
    truePostive = 0
    falseNegative = 0
    for i in range(len(gold_labels)):
        if gold_labels[i] == '1' and classified_labels[i] == '1':
            truePostive +=1
        elif gold_labels[i] == '1' and classified_labels[i] == '0':
            falseNegative +=1
    return (truePostive/(truePostive+falseNegative))

# Calculate the f1 from given gold labels and classifed labels
def f1(gold_labels, classified_labels):
    prec = precision(gold_labels, classified_labels)
    recal = recall(gold_labels, classified_labels)
    f1 = 2 * ((prec*recal)/(prec+recal))
    return f1


"""
Implement any other non-required functions here
"""

#helper function to clean the data   
def preprocessData(examples):
    clean_data = []
    for i in range(len(examples)):
        #convert lower case
        sentence = examples[i][1].lower()
        
        #remove punctuation
        sentence = re.sub('[^a-zA-Z0-9\']+', ' ', sentence)
        sentence = re.sub(r'\s+',' ', sentence)
        clean_data.append((examples[i][0], sentence, examples[i][2]))
    return clean_data

#helper function to do streaming
def tokenizer_stemming(data):
    lemmatizer = WordNetLemmatizer() 
    return [lemmatizer.lemmatize(word[0]) for word in data]

      
#helper function to generate tuples from train list
def generate_tuples_from_train(k_fold_list):
    list_word = []
    tuple_word = ()
    for idx,data in enumerate(k_fold_list):
        for doc in data:
            word = doc.split('\t')
            if len(word) == 3:
                tupleWord = (word[0],word[1],word[2])
                list_word.append(tupleWord)
    return list_word

#helper function to generate tuples from test list
def generate_tuples_from_test(testSet):
    list_word = []
    tuple_word = ()
    for word in testSet:
        word = word.split('\t')
        if len(word) == 3:
            tupleWord = (word[0],word[1],word[2])
            list_word.append(tupleWord)
    return list_word

 

#helper function for combine training data with given development and return combine datset
def createNewDataset(train_data, devData):
    train_file = open(train_data, 'r')
    dev_file = open(devData,'r')
    train_data_read = train_file.read()
    devData_read = dev_file.read()

    train_file.close()
    dev_file.close()

    train_data_read = train_data_read.split('\n')
    devData_read = devData_read.split('\n')
    combineData = train_data_read + devData_read
    combineData.pop(-1)
    return combineData

# print the recall and precision and the f1 score from k fold dataset
def print_result_score(improvedModel, k_dataset, k):
    k_dataset_copy = k_dataset
    precisionlList = []    #list for precison score
    recallList = []        #list for recall score
    f_1list = []           #list for f_1 score
    for idx in range(k):
        test = k_dataset_copy.pop(idx)
        exampleDev_text = generate_tuples_from_test(test)
        train = k_dataset_copy
        k_dataset_copy.append(test)
        exampleText = generate_tuples_from_train(train)
        
        #train each k fold
        improvedModel.train(exampleText)
        
        #get the development text from file and calculate the classify each document
        dev_list = []
        gold_labels = []
        for i in range(len(exampleDev_text)):
            dev_list.append(exampleDev_text[i][1])
            gold_labels.append(exampleDev_text[i][2])
       
        #classify for each document
        classify_labels = []
        for data in dev_list:
            classify_labels.append(improvedModel.classify(data))

        pre = precision(gold_labels, classify_labels)
        precisionlList.append(pre)
        recal = recall(gold_labels, classify_labels)
        recallList.append(recal)
        f_1 = f1(gold_labels, classify_labels)
        f_1list.append(f_1)
       
        print("SentimanetAnaysisImproved Model for k_fold {} ".format(idx+1))
        print("Recall    : {}".format(recal))
        print("Precision : {}".format(pre))
        print("F1 Score  : {}".format(f_1))

    # #calculate average k fold for precision, recall and f1
    print("***** Average for k_fold = 10 ***** ")
    print("Average of Recall    : {:.4f}".format(sum(recallList)/10.0))
    print("Average of Precision : {:.4f}".format(sum(precisionlList)/10.0))
    print("Average of F1 Score  : {:.4f}".format(sum(f_1list)/10.0))







"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:


    def __init__(self):
    # do whatever you need to do to set up your class here
        self.tupleWordPostive = []  
        self.tupleWordNegative = []
        self.NumDocu = 0          # number document
        self.NumPositive = 0      # number one class
        self.NumNegative = 0      # number zero class
        self.freqPos = {}         # frequency words in postive class
        self.freqNeg = {}         # frequency words in negative class
        self.tokenPostive = 0     # number words in postive class
        self.tokenNegative = 0    # numver words in negative class
        self.vocab = 0.0          # length of vocabulary
        self.vocab_list = []      # list of word in all document
        self.prior = {}           # calculate the probability of class
        self.likelihood = {}      # calculate the probability of word given class
        

    def train(self, examples):
        for i in range(len(examples)):
            if examples[i][2] == '1':
                self.NumPositive += 1       #count positive class
                self.tupleWordPostive.append(examples[i][1])
            else:
                self.NumNegative += 1       #count negative class
                self.tupleWordNegative.append(examples[i][1])

        
        #get the words and frequency word and store in frequency positive dictionary
        for sentence in self.tupleWordPostive:
            words = sentence.split()
            for w in words:
                if w in self.freqPos:
                    self.freqPos[w] +=1         # count positive frequency 
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
                    self.freqNeg[w] +=1      # count negative frequency 
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
        
       
        #calculate the prior of each class and add into dictionary
        self.prior['1'] = self.NumPositive/self.NumDocu
        self.prior['0'] = self.NumNegative/self.NumDocu
        
        #calcualte the likelihood of each class and add into dictionatry
        for word in self.vocab_list:

            #calculate the likelihood of positive frequency words
            if word in self.freqPos:
                likelihood_one = (self.freqPos[word] + 1) / (self.tokenPostive + self.vocab)
                self.likelihood[(word,'1')] = likelihood_one
            else: 
                likelihood_one = ( 0 + 1) / (self.tokenPostive + self.vocab)
                self.likelihood[(word,'1')] = likelihood_one

            #calculate the likelihood of negative frequency words
            if word in self.freqNeg:
                likelihood_zero = (self.freqNeg[word] + 1) / (self.tokenNegative + self.vocab)
                self.likelihood[(word,'0')] = likelihood_zero
            else:
                likelihood_zero = (0 + 1) / (self.tokenNegative + self.vocab)
                self.likelihood[(word,'0')] = likelihood_zero

    
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
                
    # classify the document as postive or negative          
    def classify(self, data):
        dict_class = self.score(data)
        if dict_class['1'] > dict_class['0']:
            return '1'
        else:
            return '0'
        
    # store feature and its value into list of tuple
    def featurize(self, data):
        words = data.split()
        word_vectors = []
        for w in words:
            word_vectors.append((w,True))
        return word_vectors
            

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

    def __init__(self):
        
        self.vocab_list = Counter()          #list of word in all document
        self.vocab_reduced = Counter()       # reduce stop words vocabulary
        self.NumClass = Counter()           # dictionary to count numbber of each class
        self.prior = {}                     #calculate the probability of class
        self.likelihood = {}                #calculate the probability of word given class
        self.freqPos = Counter()            # dictionary to count frequecny word in postive class
        self.freqNeg = Counter()            # dictionary to count frequecny word in negative class
        self.tokenPostive = 0   
        self.tokenNegative = 0
        
 
    # train our dataset   
    def train(self, examples):
       
        #get data clean
        clean_data = preprocessData(examples)

        for i in range(len(clean_data)):
            word_vectors = set(self.featurize(clean_data[i][1]))
            sentences = tokenizer_stemming(word_vectors)
            # print(sentences)
            for word in sentences:
                if not word in self.vocab_list:
                    self.vocab_list[word] = 1
                else:
                    self.vocab_list[word] +=1

        #add new feature into vocabulary and frequency postive and negivate           
        file = open("postiveWord.txt", 'r')
        file1 = open('negativeWord.txt', 'r')
        sentences = file.read()
        sentences1 = file1.read()
        file.close()
        file1.close()
        wordsListPostive = sentences.split("\n")
        wordsListNegative = sentences1.split("\n")
        
        combine_Pos_neg =  wordsListPostive + wordsListNegative
        for word in combine_Pos_neg:
            if word in self.vocab_list:
                self.vocab_list[word] += 1
            else:
                self.vocab_list[word] = 1

           
        # remove stopword
        stop = stopwords.words('english')
        for word, value in self.vocab_list.items():
            if not word in stop:
                self.vocab_reduced[word] = value
            else:
                continue
        
        # group by the class
        for i in range(len(clean_data)):
            
            #count the number class one in document
            self.NumClass['1'] += clean_data[i][2]=='1'
            
            #count the number class zero in document
            self.NumClass['0'] += clean_data[i][2]=='0'
            
            if clean_data[i][2] == '1':
                word_vectors = set(self.featurize(clean_data[i][1]))
  
                sentences = tokenizer_stemming(word_vectors)
                for word in sentences:
                    if word in self.vocab_reduced:
                        self.freqPos[word] +=1
                    else:
                        continue
            else:
                word_vectors = self.featurize(clean_data[i][1])
                sentences = set(tokenizer_stemming(word_vectors))
                for word in  sentences:
                    if word in self.vocab_reduced:
                        self.freqNeg[word] +=1
                    else:
                        continue

        #add new feature into vocabulary and frequency postive and negivate
        for word in wordsListPostive:
            if word in self.vocab_reduced:
                self.freqPos[word] +=1
            else:
                continue
        for wor in wordsListNegative:
            if word in self.vocab_reduced:
                self.freqNeg[word] +=1
            else:
                continue

       

        #get tokens word from frequecy postive 
        self.tokenPostive = sum(self.freqPos.values())
        
        #get tokens word from frequecy Negative 
        self.tokenNegative = sum(self.freqNeg.values())
        
        #number of document
        self.NumDocu = len(examples)
        
       
        #calculate the prior of each class and add into dictionary
        self.prior['1'] = self.NumClass['1']/self.NumDocu
        self.prior['0'] = self.NumClass['0']/self.NumDocu
        
        #calcualte the likelihood of each class and add into dictionatry
        for word in self.vocab_reduced:
            if word in self.freqPos:
                likelihood_one = (self.freqPos[word] + 1) / (self.tokenPostive + len(self.vocab_reduced))
                self.likelihood[(word,'1')] = likelihood_one
            else: 
                likelihood_one = ( 0 + 1) / (self.tokenPostive + len(self.vocab_reduced))
                self.likelihood[(word,'1')] = likelihood_one
            if word in self.freqNeg:
                likelihood_zero = (self.freqNeg[word] + 1) / (self.tokenNegative + len(self.vocab_reduced))
                self.likelihood[(word,'0')] = likelihood_zero
            else:
                likelihood_zero = (0 + 1) / (self.tokenNegative + len(self.vocab_reduced))
                self.likelihood[(word,'0')] = likelihood_zero
       
    
    
    # calculate the score
    def score(self, data):
        words = data.split()
        p_data_one = 0.0
        p_data_zero = 0.0
        likelihood_one = 0
        likelihood_zero = 0
        for w in words:
            if w in self.vocab_reduced:
                likelihood_one += np.log(self.likelihood[(w,'1')])
                likelihood_zero += np.log(self.likelihood[(w,'0')])
            else:
                continue
                
        p_data_one = np.exp(np.log(self.prior['1']) + likelihood_one)
        p_data_zero = np.exp(np.log(self.prior['0']) + likelihood_zero)
        
        return { '1' : p_data_one, '0' : p_data_zero}

    # classify as positive and negative
    def classify(self, data):
        dict_class = self.score(data)
        if dict_class['1'] > dict_class['0']:
            return '1'
        else:
            return '0'

    # array the word as bag of words
    def featurize(self, data):
        words = data.split()
        word_vectors = []
        for w in words:
            word_vectors.append((w,True))

        return word_vectors
            

    def __str__(self):
        return "Binary Multinomial Naive Bayes - Clean data, sentiment Lexicons, Remove stop word"

    #extra credit generate ramdom data for training and development and test
    def k_fold(self, all_examples, k):
        foldSize = int(len(all_examples)/k)
        data_split = []
        data_copy = list(all_examples)
        for idx in range(k):
            sent_split= []
            while len(sent_split) < foldSize:
                randomIndex = np.random.randint(len(data_copy))
                sentence = data_copy.pop(randomIndex)
                sent_split.append(sentence)
            data_split.append(sent_split)
        return data_split
    
 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    print(sa)
    # do the things that you need to with your base class
    exampleText = generate_tuples_from_file(training)
    sa.train(exampleText)
    
    #get the development text from file and calculate the classify each document
    exampleDev_text = generate_tuples_from_file('dev_file.txt')
    dev_list = []
    gold_labels = []
    for i in range(len(exampleDev_text)):
        dev_list.append(exampleDev_text[i][1])
        gold_labels.append(exampleDev_text[i][2])
    
   
    
    
    # classify for each document
    classify_labels = []
    for data in dev_list:
        classify_labels.append(sa.classify(data))
    print("SentimanetAnaysis Model")
    print("Recall    : {}".format(recall(gold_labels, classify_labels)))
    print("Precision : {}".format(precision(gold_labels, classify_labels)))
    print("F1 Score  : {}".format(f1(gold_labels, classify_labels)))
        
        
    
    
    improved = SentimentAnalysisImproved()
    print(improved)
    # do the things that you need to with your improved class
    improved.train(exampleText)
    
    exampleDev_text = generate_tuples_from_file('dev_file.txt')
    dev_list = []
    gold_labels = []
    for i in range(len(exampleDev_text)):
        dev_list.append(exampleDev_text[i][1])
        gold_labels.append(exampleDev_text[i][2])
    
   
    
    
    # classify for each document
    classify_labels = []
    for data in dev_list:
        classify_labels.append(improved.classify(data))
    print("SentimanetAnaysisImproved Model")
    print("Recall    : {}".format(recall(gold_labels, classify_labels)))
    print("Precision : {}".format(precision(gold_labels, classify_labels)))
    print("F1 Score  : {}".format(f1(gold_labels, classify_labels)))
    
    

    # combine train and development dataset together
    combineDataset = createNewDataset('train_file.txt', 'dev_file.txt')

    # create the 10 k fold for train and test
    k_dataset = improved.k_fold(combineDataset, 10)

    #print the recall, precision and f1 from k fold
    print_result_score(improved, k_dataset, 10)
    
