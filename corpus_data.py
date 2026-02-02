#corpus_data.py handles author list and test list
from pathlib import Path
import random
import math

#take, load the various files
def strip_header(lines: list[str]) -> list[str]:
    """
    removes headers and footers from text file
    technically you said to train on the entire data in the file if test flag, but i think header should be filtered out
    """
    #document starts after "start of" and ends after "end of", so find those lines
    #but multiple start of/end of in file

    stripped_lines = []
    valid_text = False #set to FALSE because if no header/footer exists, want it to work
    header_footer_exists = False

    #loop through lines, if header_footer is FALSE then add to stripped_lines
    for line in lines:
        if "*** START OF" in line:
            valid_text = True
            header_footer_exists = True
        elif "*** END OF" in line:
            valid_text = False
        elif valid_text == True:
            stripped_lines.append(line)

    #if there aren't any *** START / **** END still want to return
    if header_footer_exists == False: #if header doesn't exist
        return lines #return everything, nothing to get rid of
    else:
        return stripped_lines
    
def remove_empty_lines(lines: list[str]) -> list[str]:
    """
    If line is empty, removes them, so don't end up with empty liens in train/test
    """
    non_empty_lines = []
    for line in lines:
        line = line.strip()
        if line != "":
            non_empty_lines.append(line)

    return non_empty_lines


def load_training_set(authorlist_arg:str) -> dict[str, Path]:
    """
    Takes in authorlist argument from arg parser, and returns file paths, with author names
    """
    #grab file names and author names
    path_auth_name_dict = {}
    
    #this will break if author puts in "authorlist" and file name is "authorlist.txt"
    author_path = Path(authorlist_arg)

    lines = author_path.read_text().splitlines()

    #then extract out the path, and the author name
    for auth_file in lines:
        #clear white space
        auth_file = auth_file.strip()

        #get parent file path of authorlist, and then append auth_file, this is assuming no subfolders
        train_auth_path = author_path.parent/auth_file

        #grabbing part before underscore, assuming authors name is one word not like j_r_r_tolkien or something
        auth_name = auth_file.split('_', 1)[0]

        #now which is the author, which is the key? 
        #should we assume author is unique?
        path_auth_name_dict[auth_name] = train_auth_path

    return path_auth_name_dict

#if test flag, read in test file
def load_train_test_set(train_corpus_dict: dict[str, Path], testfile_arg: str) -> tuple[dict[str, list[str]], list[str]]: 
    """
    takes in the training set, keeps entire set
    takes in test set, outputs back
    """

    train_set_dict = {}

    for author, auth_path in train_corpus_dict.items():
        text = auth_path.read_text().splitlines()
        text = strip_header(text)
        text = remove_empty_lines(text)

        train_set_dict[author] = text #keep entire text as train set

    #will need to read in testfile, then have to output per line, so return as splitlines()?
    test_set_path = Path(testfile_arg)

    test_lines = test_set_path.read_text().splitlines()
    #can the test lines have a header?? probably not

    return train_set_dict, test_lines

def dev_train_split(train_corpus_dict: dict[str, Path]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:###ADD TYPEHINT HERE###: #maybe have return train and test split?
    """
    90% train, 10% dev split
    returns a tuple of two dictionaries, one with author: training_set, the other with author: dev_set
    """
    random.seed(999) #set random seed = reproducible
    dev_set_dict = {}
    train_set_dict = {}

    for author, auth_path in train_corpus_dict.items(): 
        text = auth_path.read_text().splitlines()
        text = strip_header(text) #strip headers, handles if header exists or does not exist
        text = remove_empty_lines(text) #remove empty lines

        random.shuffle(text) #this shuffling shouldn't be a huge problem? 
        #it does mix up sentences but if we treat end of line as sentence

        num_lines = len(text)
        split_num = num_lines*0.9 #find 90th line to cut on
        split_num = math.floor(split_num) #floor so have integer

        #split into train/test
        train_set = text[:split_num]
        dev_set = text[split_num:]

        #add to dict, assuming no duplicate authors
        dev_set_dict[author] = dev_set
        train_set_dict[author] = train_set


    return train_set_dict, dev_set_dict


#Note: do i want to add start of sentence identifiers or other things. or remove \n

##deal with empty lines?
