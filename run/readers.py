from __future__ import print_function
from spacy.en import English
import os 
import io

def check_pronoun(word):
    if word == 'I' or word == 'me' or word == 'we' or word == 'us':
        return True
    elif word == 'he' or word == 'she' or word == 'her' or word == 'him':
        return True
    elif word == 'they' or word == 'them':
        return True
    elif word == 'myself' or word == 'ourselves' or word == 'yourself' or word == 'himself' or word == 'herself':
        return True
    elif word == 'Me' or word == 'We' or word == 'Us':
        return True
    elif word == 'He' or word == 'She' or word == 'Her' or word == 'Him':
        return True
    elif word == 'They' or word == 'Them':
        return True
    elif word == 'Myself' or word == 'Ourselves' or word == 'Yourself' or word == 'Himself' or word == 'Herself':
        return True
    else:
        return False

def end_of_sentence(word):
    if word == '.' or word == '!' or word == '?':
        return True
    else:
        return False

"""
These functions read in all the books from path_file and returns them in a dictionary.

Path file format:
book_id [tab] /path/of/file

Dictionary format:
{book_id1, [('list', pronoun_count, sentence_count), ...], ... }
"""

def read_a_book(file):
    words = []

    pronoun_count = 0
    sentence_count = 0

    for line in open(file, 'r'):
        splitted = line.split()

        for word in splitted:
            if (check_pronoun(word)):
                pronoun_count += 1
            if (end_of_sentence(word)):
                sentence_count += 1

            words.append((word, pronoun_count, sentence_count))

    return words

def read_normal_books(path_file):
    paths = [word for line in open(path_file, 'r') for word in line.split()]
    path_dict = dict([(k, v) for k,v in zip (paths[::2], paths[1::2])])

    book_dict = {}

    for key in path_dict:
        book_dict[key] = read_a_book(path_dict[key])

    return book_dict

def read_x_books(path_file):
    paths = [word for line in open(path_file, 'r') for word in line.split()]
    path_dict = dict([(k, v) for k,v in zip (paths[::2], paths[1::2])])

    book_dict = {}

    for key in path_dict:
        words = []

        pronoun_count = 0
        sentence_count = 0
        
        for line in open(path_dict[key], 'r'):
            splitted = line.split()
            word = splitted[7]

            if (check_pronoun(word)):
                pronoun_count += 1
            if (end_of_sentence(word)):
                sentence_count += 1

            words.append((word, pronoun_count, sentence_count))

        book_dict[key] = words

    return book_dict

"""
Creates an index for quick look-up between
data annotated in (line, pos) format and their
absolute position in the book
"""
def create_lined_index(file):
    lined_book = []

    index = 0

    for i, line in enumerate(open(file, "r")):
        temp_line = []

        for word in line.split():
            if not word == "": 
                temp_line.append(index)
                index += 1

        lined_book.append(temp_line)

    return lined_book

"""
These functions read in testing and training data and returns them in lists of data points

Training and testing set format:
[(book_id, anaphor1, antecedent1), ... ]

"""
def read_data(training_file, training_ratio):
    train = []
    test = []

    training_ratio *= 10
    count = 0

    #using with open() as file: instead?
    for line in open(training_file, 'r'):
        splitted = line.split()
        datum = (splitted[0], int(splitted[1]), int(splitted[2]))

        if count < training_ratio:
            train.append(datum)
        else:
            test.append(datum)
            if count >= 10:
                count = 0

        count += 1

    return train, test

def read_lined_data(training_path, training_ratio, path_path, book_key):
    train = []
    test = []

    training_ratio *= 10
    count = 0

    for line in open(training_path, 'r'):
        datum = ()

        splitted = line.split("-")
        first = splitted[0].strip().split(",")
        second = splitted[1].strip().split(",")

        first_line = first[0].strip("(")
        first_pos = first[1].strip(")")

        second_line = second[0].strip("(")
        second_pos = second[1].strip(")")

        book_path = [line.split()[1] for line in open(path_path, 'r')]

        index = create_lined_index(book_path[0])

        datum = (book_key, index[int(first_line)][int(first_pos)], index[int(second_line)][int(second_pos)])

        if count < training_ratio: 
            train.append(datum)
        else:
            test.append(datum)
            if count >= 10:
                count = 0

        count += 1
    return train, test