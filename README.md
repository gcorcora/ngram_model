# ngram_model
NGram model for CSCI5541 assignment

Uses a trigram model. Can be edited for bigram, or other gram models.

Will require associated training corpora listed in an "authorlist" file.
E.g., within the file are a list of training files labeled things like below, and will create a separate model for each author.
"[author]_train.txt"
"[author1]_train.txt"

To run with a training and dev set (10% of sentences extracted from dev set for each training file), use command:

classifier.py authorlist 

The output will return the results on the devset, and the % correct for each author.

One can supply a test set, and the entire training corpora will be used to trian the models. 

Use this command:

python3 classifier.py authorlist -test [author_test_sentences_file].txt

Assignment writeup included in csci5541_hw1_corcoran.pdf

