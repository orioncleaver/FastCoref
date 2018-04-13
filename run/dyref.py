from __future__ import print_function
import argparse
import dynet as dy
import numpy as np
import random

import time

#personal files
import readers

LOOKBACK_LENGTH = 100
TRAINING_RATIO = 0.8

class CorefModel:
    def __init__(self): 
        self.books = { }
        self.training_data = { }
        self.testing_data = { } 
        self.feature_store = [ ]

    def read_books(self, format_, args):
        """
        This function reads in all the books from path_path and returns them in a dictionary.
        It also reads all the training data and splits it into two arrays, training and testing.
    
        Path file format:
        book_key [tab] /path/of/file

        Returned book dictionary format:
        books = {book_key1, ['list', 'of', 'all', 'words'], ... }
        
        Return data array format: 
        training_data = [ (book_id, anaphor, antedecent), ... ]
        testing_data = [ (book_id, anaphor, antedecent), ... ]
        
        S annotation format: 
        (row_of_anaphor, col_of_anaphor), (row_of_antedecent, col_of_antedecent

        X annotation format:
        book_id [tab] anaphor [tab] antedecent

        O annotation format:
        In-text annotation
        """

        if format_ == 's':
            self.books = readers.read_normal_books(args.books)
            # only handles one book for now
            key = list(self.books.keys())[0]
            self.training_data, self.testing_data = readers.read_lined_data(args.data_file, TRAINING_RATIO, args.books, key)

        if format_ == 'x':
            self.books = readers.read_x_books(args.books)
            self.training_data, self.testing_data = readers.read_data(args.data_file, TRAINING_RATIO)
        
        #sif format_ == 'o':
            

    def get_common_features(self, book, anaphor, antecedent, is_training, nn):
        features = []
        features.append(book[antecedent - 1][0])
        features.append(book[antecedent][0])
        features.append(book[antecedent + 1][0])

        features.append(book[anaphor - 1][0])
        features.append(book[anaphor][0])
        
        features.append("dist_" + str(anaphor - antecedent))
        features.append("pronouns_" + str(book[anaphor][1] - book[antecedent][1]))
        features.append("sentences_" + str(book[anaphor][2] - book[antecedent][2]))

        if is_training and random.randint(0, 2499) == 1:
            features.append("_UNK_")
        elif nn == False or (book[anaphor][0] + "_" + book[antecedent][0]) in nn.w2i:
            features.append(book[anaphor][0] + "_" + book[antecedent][0])
        else:
            features.append("_UNK_")

        return features

class LinearModel (CorefModel): 
    W = {}

    def __init__(self):
        CorefModel.__init__(self)

    def get_features(self, book, anaphor, antecedent, is_training):
    	return self.get_common_features(book, anaphor, antecedent, is_training, False)

    def change_weight(self, features, delta):
        for f in features:
            if f not in self.W:
                self.W[f] = 0
            self.W[f] += delta

    def get_best(self, anaphor, book):
        best = [0, 0]
                
        for i in range(LOOKBACK_LENGTH):
            a = anaphor - i 

            if a >= 0:
            	features = self.get_features(book, anaphor, a, True)

            	score = self.get_score(features)

            	if score > best[0]:
                	best[0] = score
                	best[1] = a

        return best
    
    def get_score(self, features):
        score = 0
        for f in features:
            if f in self.W: 
                score += self.W[f]
            else:
                self.W[f] = 0
        return score

    def train(self, v):
        correct_num = 0
        total = len(self.training_data)

        for progress, datum in enumerate(self.training_data):
            best = self.get_best(datum[1], self.books[datum[0]])

            if (best[1] == datum[2]):
                correct_num += 1
            else:
                features = self.get_features(self.books[datum[0]], datum[1], datum[2], True)
                self.change_weight(features, 1)
                features = self.get_features(self.books[datum[0]], datum[1], best[1], True)
                self.change_weight(features, -1)

        print(" accuracy: " + str(correct_num) + "/" + str(total))

    def test(self, v):
        total = len(self.testing_data)
        correct_num = 0

        for counter, datum in enumerate(self.testing_data):
            best = self.get_best(datum[1], self.books[datum[0]])

            if v:
            	anaph = self.books[datum[0]][datum[1]][0]
                antec = self.books[datum[0]][datum[2]][0]
            	print(str(counter - 600) + ": Match found! " + anaph + " at " + str(datum[1]) + " matches " + self.books[datum[0]][best[1]][0] + " at " + str(best[1]))

            if best[1] == datum[2]:
                correct_num += 1

                if v:
                    print("ACTUAL MATCH: " + self.books[datum[0]][best[1]][0] + " is the same as " + antec + " at " + str(best[1]) + " and " + str(datum[2]))
                    print(" ")
                    print(" ")
            elif v: 
                print("NOT ACTUAL MATCH: " + self.books[datum[0]][best[1]][0] + " is different than " + antec + " at " + str(best[1]) + " and " + str(datum[2]))


        print("")
        print("on the testing set:")
        print("  accuracy: " + str(correct_num) + "/" + str(total))


        return correct_num

    def run(self, v, run_file):
        run_book = readers.read_a_book(run_file)

        for index, word in enumerate(run_book):
            best = self.get_best(index, run_book)

class NeuralNet (CorefModel):
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)

    pW = model.add_parameters((32, 64))
    pb = model.add_parameters(32)

    pV = model.add_parameters((1, 32))
    pc = model.add_parameters(1)

    w2i={}
    words=[]

    def __init__(self, clustering):
        CorefModel.__init__(self) 
        if clustering:
            self.clustering = True
        else:  
            self.clustering = False

    def get_features(self, book, anaphor, antecedent, is_training):
    	embs = []

    	features = self.get_common_features(book, anaphor, antecedent, is_training, self)

    	for f in features:
            embs.append(self.E[self.w2i[f]])

        if (self.clustering):
    	    cluster = []
    	    for i in range(10):
    		   cluster.append(self.E[self.w2i[book[antecedent - (i + 1)][0]]])
    		   cluster.append(self.E[self.w2i[book[antecedent + (i + 1)][0]]])

    	    embs.append(dy.average(cluster))

    	    cluster2 = []
    	    index = anaphor
    	    word = book[anaphor][0]
    	    while index > 0 and (word != '.' and word != '?' and word != '!'):
    	   	    cluster2.append(self.E[self.w2i[word]])
    	   	    index -= 1
    		    word = book[index][0]

            index = anaphor + 1
            word = book[index][0]

            while index < (len(book) - 1) and (word != '.' and word != '?' and word != '!'):
        		cluster2.append(self.E[self.w2i[word]])
        		index += 1
        		word = book[index][0]

            embs.append(dy.average(cluster2))

            cluster3 = []
            index = antecedent
            word = book[antecedent][0]
            while index > 0 and (word != '.' and word != '?' and word != '!'):
                cluster3.append(self.E[self.w2i[word]])
                index -= 1
                word = book[index][0]

            index = antecedent + 1
            word = book[index][0]

            while index < (len(book) - 1) and (word != '.' and word != '?' and word != '!'):
                cluster3.append(self.E[self.w2i[word]])
                index += 1
                word = book[index][0]

            embs.append(dy.average(cluster3))

    	return embs

    def init_after_read(self):
        self.add_words()
        self.words_num = len(self.words)
        self.E = self.model.add_lookup_parameters((self.words_num * 2, 64))

    def add_word(self, w):
        if w in self.w2i:
            return
        n = len(self.words)
        self.words.append(w)
        self.w2i[w] = n

    def add_words(self):
        for book in self.books:
            for word in self.books[book]:
                self.add_word(word[0])

        for n in range(LOOKBACK_LENGTH + 1):
            self.add_word("dist_" + str(n))
            self.add_word("pronouns_" + str(n))
            self.add_word("sentences_" + str(n))

        for datum in self.training_data:
            for i in range(LOOKBACK_LENGTH):
                a = datum[1] - i
                self.add_word(self.books[datum[0]][datum[1]][0] + "_" + self.books[datum[0]][a][0])
        self.add_word("_UNK_")
        self.add_word("range_detector_dummy_val")

    def get_correct_index(self, anaphor, antecedent):
        if (antecedent > anaphor):
            print("ERROR! Anaphor preceeding antecedent in training/testing data")

        i = anaphor - antecedent + 1

        if i <= LOOKBACK_LENGTH:
            return i
        else: 
            return 0

    def get_scores(self, book, anaphor, is_training):
        dy.renew_cg()
        W = dy.parameter(self.pW)
        V = dy.parameter(self.pV)
        b = dy.parameter(self.pb)
        c = dy.parameter(self.pc)

        I = []

        #add out-of-range to end of range
        embs = []
        embs.append(self.E[self.w2i["range_detector_dummy_val"]])
        embs.append(self.E[self.w2i["dist_" + str(LOOKBACK_LENGTH)]])
        embs.append(self.E[self.w2i["pronouns_" + str(LOOKBACK_LENGTH)]])
        embs.append(self.E[self.w2i["sentences_" + str(LOOKBACK_LENGTH)]])
        layer_1 = dy.esum(embs)
        layer_2 = dy.rectify(W * layer_1 + b)
        output = dy.tanh(V * layer_2 + c)
        I.append(output)

        for i in range(LOOKBACK_LENGTH):
            a = anaphor - i
            
            if a >= 0: 
            	embs = self.get_features(book, anaphor, a, is_training)
                
            	layer_1 = dy.esum(embs)
            	layer_2 = dy.rectify(W * layer_1 + b)
            	output = dy.tanh(V * layer_2 + c)
            	I.append(output)

        return dy.concatenate(I)

    def train(self, v):
        total = len(self.training_data)
        total_loss = 0
        correct_num = 0
        
        random.shuffle(self.training_data)
        for datum in self.training_data:
            book = self.books[datum[0]]

            scores = self.get_scores(book, datum[1], True)
            
            correct_index = self.get_correct_index(datum[1], datum[2])

            guess_index = np.argmax(dy.softmax(scores).npvalue())
            if correct_index == guess_index:
                correct_num += 1
                if v:
                    if correct_index == 0:
                        print("MATCH! Correctly guessed out-of-range")
                    elif correct_index == 1:
                        print("MATCH! Correctly guessed self-link")
                    else:
                        print("MATCH! Correct index: " + str(correct_index))
                        #print("  Guessed Word: ", book[datum[1] - guess_index])
                        #print("  Correct Word: ", book[datum[1] - correct_index])
            elif v:
                if correct_index == 0:
                    print("WRONG! Failed to guess out-of-range")
                    print("WRONG! Correct dex: " + str(correct_index))
                elif guess_index == 0:
                    print("WRONG! Incorrectly guessed out-of-range")
                    print("WRONG! Correct dex: " + str(correct_index))
                elif correct_index == 1:
                    print("WRONG! Failed to guess self-link")
                    print("WRONG! Correct dex: " + str(correct_index))
                elif guess_index == 1:
                    print("WRONG! Incorrectly guessed self-link")
                    print("WRONG! Correct dex: " + str(correct_index))
                else:
                    print("WRONG! Guess index: " + str(guess_index))
                    print("WRONG! Correct dex: " + str(correct_index))

            loss = dy.pickneglogsoftmax(scores, correct_index)
            total_loss += loss.value()
            loss.backward()
            self.trainer.update()

        print(" loss: " + str(int(total_loss)))
        print(" accuracy: " + str(correct_num) + "/" + str(total))

    def test(self, v): 
        total = len(self.testing_data)
        correct_num = 0
        
        for datum in self.testing_data:
            book = self.books[datum[0]]

            scores = self.get_scores(book, datum[1], False)

            if self.get_correct_index(datum[1], datum[2]) == np.argmax(dy.softmax(scores).npvalue()):
                correct_num += 1
        print("")
        print("on the testing set:")
        print("  accuracy: " + str(correct_num) + "/" + str(total))

    def run(self, v, run_file):
        run_book = readers.read_a_book(run_file)

        for word in run_book:
            self.add_word(word)

        for index, word in enumerate(run_book):
            if index + 1 < len(run_book):
                scores = self.get_scores(run_book, index, False)

                index_guess = np.argmax(dy.softmax(scores).npvalue())


def main(args):
    format_ = ''

    if args.s:
        format_ = 's'
    if args.x:
        format_ = 'x'
    if args.o:
        format_ = 'o'

    if args.linear or args.all:
        lm = LinearModel()
        lm.read_books(format_, args)

        print("Training in Progress")
        for ITER in range(10):
            print("  iteration " + str(ITER) + ":")
            lm.train(args.v)
        
        if not args.run:
            lm.test(args.v)
        else:
            print("Running in Progress")
            start = time.time()
            lm.run(args.v, args.run)
            end = time.time()
            print(end - start)

    if not args.linear:
        nn = NeuralNet(args.cluster)
        nn.read_books(format_, args)
        nn.init_after_read()

        print("Training in Progress")
        for ITER in range(10):
            print("  iteration " + str(ITER) + ":")
            nn.train(args.v)
        
        if not args.run:
            nn.test(args.v)
        else:
            print("Running in Progress")
            start = time.time()
            nn.run(args.v, args.run)
            end = time.time()
            print(end - start)

if __name__ == "__main__":
    verbose = False
    parser = argparse.ArgumentParser(description='Machine learning coreference resolution model')
    parser.add_argument('--v', action='store_true', default=False, help='increase output verbosity')
    parser.add_argument('--linear', action='store_true', default=False, help='runs linear model instead of neural model')
    parser.add_argument('--all', action='store_true', default=False, help='runs both models, neural runs alone by default')
    parser.add_argument('--cluster', action='store_true', default=False, help='runs neural model with clustering')
    parser.add_argument('--s', action='store_true', default=False, help='format of data (choose one)')
    parser.add_argument('--x', action='store_true', default=False, help='format of data (choose one)')
    parser.add_argument('--o', action='store_true', default=False, help='format of data (choose one)')
    parser.add_argument("data_file", type=str, default='data/alicesadventures.txt.annotation', help='path to data annotated in the chosen format')
    parser.add_argument('--books', type=str, default='paths/alicePath.txt', help='list of books in format ["custom_key" "book_path"]')
    parser.add_argument('--run', type=str, default=False, help='parses coreference for entire file')
    args = parser.parse_args()
    main(args)
