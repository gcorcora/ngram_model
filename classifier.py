#n-gram LM to classify authors
#this is master file

import tiktoken
import argparse #so can do -test
import corpus_data

#use tiktoken with o200k encoding

#command to run needs to be python3 classifier.py authorlist -test testfile and 
#python3 classifer.py authorlist

enc=tiktoken.get_encoding("o200k_base")

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
    train_corpus_dict = corpus_data.load_training_set(args.authorlist) 

    
    


    #read in testfile, if applicable




if __name__ == "__main__":
    main()