#corpus_data.py handles author list and test list
from pathlib import Path

#take, load the various files
def load_training_set(authorlist_arg:str) -> dict[str, Path]:
    """
    Takes in authorlist argument from arg parser, and returns a list of file paths, with author names
    """
    #grab file names and author names
    path_auth_name_dict = {}
    
    #this is entirely assuming file is just "authorlist" and not "authorlist.txt" without .txt specified
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


#split into train_test if no -test flag

#if test flag, read in test file