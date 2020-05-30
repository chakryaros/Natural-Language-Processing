from collections import Counter
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import sys


class LanguageModel:
    def __init__(self, n_gram, is_laplace_smoothing, backoff=None):
        self.Ngram = n_gram
        self.freq = None
        self.word = None
        self.Numtokens = None
        self.smoothing = is_laplace_smoothing
        self.content = None
        self.bigram = {}
        self.unigramProb = {} # unigram probability dic
        self.bigramProb = {} # bigram probability dict
        
    def train(self, training_file_path):
        f = open(training_file_path, "r")
        self.content = f.read()
        f.close()
        self.word = self.content.split()
       
        #count the frequency word in dictionary
        self.freq = Counter(self.word)
       
       
        #assign <UNK> to the frequecy word that less than 2
        for key, value in list(self.freq.items()):
            if value == 1:
                self.freq['<UNK>'] += 1
                del self.freq[key]

        # bigram frequecy word        
        for i in range(len(self.word)-1):
            if self.word[i] not in list(self.freq):  #check if the first word not in frequency
                self.word[i] = '<UNK>'
            if self.word[i+1] not in list(self.freq): #check if the next word not in frequency
                self.word[i+1] = '<UNK>'
            if (self.word[i],self.word[i+1]) not in self.bigram: #check if the first and next word not in frequency
                self.bigram[(self.word[i],self.word[i+1])] = 1
            else:
                self.bigram[(self.word[i],self.word[i+1])] += 1
        
            #calculate probability of bigram words for generate sentence
            self.bigramProb[(self.word[i],self.word[i+1])] = self.bigram[(self.word[i],self.word[i+1])]/sum(self.bigram.values())

        #count the number of tokens
        self.Numtokens = sum(self.freq.values())
       
        
        #calculate probability of each word for generate sentence
        for word in self.word:
            self.unigramProb[word] = self.freq[word]/sum(self.freq.values())
         
        
        

    #helper function for calculate the bigram probability for each bigram
    def bigram_probability(self,bigram):
        prob = 0.0
        count_word = 0.0
        bottom = 0.0
        if self.smoothing:
            for i in range(len(bigram)-1):
                if (bigram[i], bigram[i+1]) not in self.bigram: #check if bigram word not in bigram frequecy
                    count_word = 1.0     #count word zero, just add_smooth
                else:
                    #if self_bigram has bigram word, get the values and add_smooth
                    count_word = self.bigram[bigram[i],bigram[i+1]] + 1
                if bigram[i] not in self.freq:  #check for single word
                    bottom = 1.0
                else:
                    bottom = self.freq[bigram[i]] + len(self.freq)
            prob = count_word/bottom
        else:

            for i in range(len(bigram)-1):
                if (bigram[i], bigram[i+1]) not in self.bigram: #check if bigram word not in bigram frequecy
                    prob = 0.0
                else:
                    prob = self.bigram[bigram]/self.freq[bigram[0]]
            
            # prob = self.bigram[bigram]/(self.freq[bigram[0]])
        return prob
       
    #helper function for calculate the unigram probability for each word
    def unigram_probability(self,word):
        prob = 0.0
        if self.smoothing:
            prob = (self.freq[word] + 1)/(self.Numtokens + len(self.freq))
           
        else:
            prob = (self.freq[word])/(self.Numtokens)
        return prob  
        
    #helper function for generate unigram random
    def unigram_generate(self):
        start = '<s>'
        word_freqs = []
        word_freqs.append(start)
      
        #check if the word is not the end of sentence
        while start != '</s>':
            unigram_freqs = {}
            #loop through frequecy words
            for w in list(self.freq.keys()):
                if w != '<s>':
                    unigram_freqs[w] = self.unigramProb[w]  #calcutate the probability word
            total = sum(unigram_freqs.values())    # calcuate the total of frequence word values   
            for i in list(unigram_freqs.keys()):
                unigram_freqs[i] = unigram_freqs[i]/total
            #generate the random word in the range of probability
            random_word = random.choices(list(unigram_freqs.keys()), list(unigram_freqs.values()))
            word_freqs.append(random_word[0])
            while word_freqs[1] == '</s>':
                word_freqs.pop()
                random_word = random.choices(list(unigram_freqs.keys()), list(unigram_freqs.values()))
                word_freqs.append(random_word[0])
            start = random_word[0]
        word_freqs = ' '.join(word_freqs)
        return word_freqs
    
    def bigram_genterate(self):
        start = '<s>'
        sentence = []
        sentence.append('<s>')
        #check if the word is not the end of sentence
        while start != '</s>':
            bigram_freqs = {}
            #loop through frequecy words
            for word in list(self.bigramProb.keys()):
                if word[0] == start:
                    #calcutate the probability bigram word
                    bigram_freqs[word] = self.bigramProb[word]

#             total = sum(bigram_freqs.values())
#             for word in bigram_freqs:
#                 bigram_freqs[word] = bigram_freqs[word]/total
            #random select word from bigram frequence
            keys=np.array(list(bigram_freqs.keys()))
            prob=np.array(list(bigram_freqs.values()))
            prob/= prob.sum()
            index = np.random.choice(len(keys),1,p=prob)
            word=keys[index]
            word=keys[np.random.choice(len(keys),1,p=prob)]
            sentence.append(word[0][1])
            start = word[0][1]

        sentence = " ".join(sentence)
        return sentence
   
            
    # generate the random sentence
    def generate(self, num_sentences):
        arr_sentences = []
        sentence = ''
        for _ in range(num_sentences):
            if self.Ngram == 1:
                sentence = self.unigram_generate() #generate the unigram sentence
                arr_sentences.append(sentence)
            else:
                sentence = self.bigram_genterate() #generate the bigram sentence
                arr_sentences.append(sentence)
        return arr_sentences
            

       
    #calcuate the probability of each sentence
    def score(self, sentence):
        prob =1.0
        sentence = sentence.split()
        #calculate the probability of unigram
        if self.Ngram == 1:
            for i in range(len(sentence)):
                if sentence[i] not in self.freq:
                    sentence[i] = '<UNK>'
            for w in sentence:
                prob = prob * self.unigram_probability(w)
                
        #calculate the probability of bigram
        else:
            for i in range(len(sentence)-1):
                if sentence[i] not in self.freq:
                    sentence[i] = '<UNK>'
                if sentence[i+1] not in self.freq:
                    sentence[i+1] = '<UNK>'
                sentence[i] = (sentence[i],sentence[i+1])
            sentence.pop(-1)
            for w in sentence:
                prob = prob * self.bigram_probability(w)
        return prob
    
    #helper function to write probabiliy to file
    def prob_to_file(self, test_file, outfile):
        file = open(test_file)
        #read each line of file
        lines = file.readlines()
        lines.pop(-1)
        #open file to write probability to file
        out = open(outfile, 'w')
        for sentence in lines:
            #caculate the probability of unigram
            prob = self.score(sentence)
            out.write(str(prob) +"\n")
        file.close()
        out.close()
    
    #plot histogram for test set and my test set
    def plot_histogram(self, test_file, my_test_set, savefile, Ngram):
        hw_file_test = open(test_file)
        sentence_test = hw_file_test.readlines()
        prob_hw_test = []
        sentence_test.pop(-1)
        for sentence in sentence_test:
            prob_hw_test.append(self.score(sentence))
        
        myFile = open(my_test_set)
        sentence_mytest = myFile.readlines()
        prob_my_test = []
        sentence_mytest.pop(-1)
        for s in sentence_mytest:
            prob_my_test.append(self.score(s))
        hw_file_test.close()
        myFile.close()
        concate_list = prob_hw_test + prob_my_test
        overall_min = min(concate_list)
        #plot the histogram
        min_exponent = np.floor(np.log10(np.abs(overall_min)))
        plt.hist([prob_hw_test,prob_my_test],bins=np.logspace(np.log10(10**min_exponent),np.log10(1.0)), label = ["hw2-test", "My test set"], stacked = True) 
        
        plt.xscale('log')
        plt.title("The relative frequency of the probabilities of the test set " + Ngram)

        plt.xlabel("Probability of test set")
        plt.ylabel("Frequecy")
        plt.legend()
        plt.savefig(savefile,bbox_inches='tight')
        plt.show()
        
    #extra credit to calculate the perplexity
    def perplexity(self,test_sequence):
        file = open(test_sequence)
        word = file.read()
        sentences = file.readlines()
        word = word.split()
        NumWord = len(word)
        per_log_sum = 0
        perplex = 0
        for s in sentences:
            per_log_sum -= math.log(self.score(s),2)
        perplex = math.pow(2, (per_log_sum/NumWord))
        return perplex
            
           
                
        
def main(args):
    
    #Unigram model
    training_file_path = args[1] #for berp-training.txt
    hw2_test_file = args[2]     #for hw2-test.txt
    hw2_my_test_file = args[3]  #for hw2-my-test.txt
    LM = LanguageModel(1, True, backoff=None)
    LM.train(training_file_path)
    random_sentences = LM.generate(100)
    generate_file = open("hw2-unigram-generated.txt", 'w')
    for sentence in random_sentences:
        generate_file.write(str(sentence) + '\n')
    generate_file.close()
    LM.prob_to_file(hw2_test_file,"hw2-unigram-out.txt")
    LM.plot_histogram(hw2_test_file, hw2_my_test_file, "hw2-unigram-histogram.pdf", "Unigram")

    
    #bigram model
    LM_bigram = LanguageModel(2, True, backoff=None)
    LM_bigram.train(training_file_path)
    random_bigram = LM_bigram.generate(100)
    generate_bigram = open("hw2-bigram-generated.txt", 'w')
    for sentence in random_bigram:
        generate_bigram.write(str(sentence) + '\n')
    generate_bigram.close()
    LM_bigram.prob_to_file(hw2_test_file,"hw2-bigram-out.txt")
    LM_bigram.plot_histogram(hw2_test_file, hw2_my_test_file, "hw2-bigram-histogram.pdf", "Bigram")
   
        
        

                             
                
if __name__ == "__main__":
    
    main(sys.argv)
    
