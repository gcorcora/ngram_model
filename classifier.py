#n-gram LM to classify authors
#this is master file

import argparse #so can do -test
import corpus_data as c
import tokenizer as tk

#use tiktoken with o200k encoding

#command to run needs to be python3 classifier.py authorlist -test testfile and 
#python3 classifer.py authorlist

def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="classifier",
                                     description = "Classify authors based on a set of training text")
    
    #author list is file containing a list of file names like austen_train.txt
    parser.add_argument("authorlist")

    #if no test, grab 10% devset, train on remainder. then run task on devset, print out result
    #if test, use all data, for each author file, then output results for each **LINE** in test file
    #each line is entire sentence
    parser.add_argument("-test", dest="testfile", default=None)

    args = parser.parse_args()

    return args

def main() -> None:
    args = argument_parser()

    #read in author list, get a dict with the path and the author name
    train_corpus_dict = c.load_training_set(args.authorlist) 

    #read in testfile, if applicable
    if args.testfile is None:
        #if no test file, must take 10% dev set
        train_set_dict, dev_set_dict = c.dev_train_split(train_corpus_dict)
        
        #tokenize + train, returning n-gram model for each author
        #set desired n-gram length, too if wanted
        ngram_len = 3
        #this returns back a dict of author: {ngram_len: (ngram_count_dict, context_count_dict}
        models = tk.train_ngram_model(train_set_dict, ngram_len)

        #now classify based on dev set, print result
        #do stupid backoff, calculate perplexity for each line in dev set per author model
        #set up fn in tokenizer to do this
        tk.dev_test_results(ngram_len, models, dev_set_dict)

    else:
        #train, then test on lines
        #load train and test set
        train_set_dict, test_lines = c.load_train_test_set(train_corpus_dict, args.testfile)

        ngram_len = 3

        #train ngram models
        models = tk.train_ngram_model(train_set_dict, ngram_len)

        #now classify each line in test_lines based on models
        tk.test_file_results(ngram_len, models, test_lines)




if __name__ == "__main__":
    main()
