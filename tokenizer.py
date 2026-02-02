#this file handles the tokenizer?
import tiktoken
import math

def tokenize_text(lines_to_tokenize: list[str], encoder: tiktoken.Encoding, n:int = 2) -> list[list[int]]:
    tokenized_lines = []

    for line in lines_to_tokenize:
        #this doesn't /really/ have to be an if statement, but it makes it clearer what is happening
        #so i kept it this way

        #using negative numbers because tiktoken ints are greater than zero
        if n==1:
            token_line = encoder.encode(line) + [-2] #if unigram, don't care about start 
        elif n == 2:
            token_line = [-1] + encoder.encode(line) + [-2] #list of integers. [-1] is start token, [-2] is end token
        else:
            #n > 2, so we add more start tokens, keep only one end token
            start_tokens = [-1] * (n - 1)
            token_line = start_tokens + encoder.encode(line) + [-2]
        tokenized_lines.append(token_line)

    return tokenized_lines

def n_gram_context_maker(text: list[list[int]], ngram_len: int) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]:
    #returns two dictionaries, one for context and one for ngram count
    ngram_count = {}
    context_count = {}
    
    for line in text: #already tokenized
        #go length of line, collecting each ngram
        for i in range(len(line) - ngram_len + 1):
            #at each i, grab ngram
            ngram = line[i: i+ngram_len] #slice of length ngram
            #but slice of list is not what need
            ngram = tuple(ngram) #make into tuple so can be dict key

            #context is first part of ngram
            context = ngram[0:-1]
            if ngram in ngram_count:
                ngram_count[ngram] += 1
            else:
                ngram_count[ngram] = 1
            if context in context_count: 
                context_count[context] += 1
            else:
                context_count[context] = 1

    return ngram_count, context_count

def train_ngram_model(set_to_train: dict[str, list[str]], ngram_len:int = 3) -> dict[str, dict[int, tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]]]: #will eventually have it return each model
    #returns a dict of author: (ngram_count_dict, context_count_dict)
    print("training n-gram models... (this may take awhile)")
    enc = tiktoken.get_encoding("o200k_base")

    models = {}

    for author, text in set_to_train.items():
        #tokenize then build ngram model for each
        #next find frequencies of each ngram, and frequencies of context
        models_backoff = {} #create to store backoff models that may be needed
        for n in range(1, ngram_len + 1): #build ngram models up to desired len in case backoff needed
            text_tokens_list = tokenize_text(text, enc, n)
            ngram_count, context_counts = n_gram_context_maker(text_tokens_list, n) 
            models_backoff[n] = (ngram_count, context_counts)
        models[author] = models_backoff #so each author maps to dict itself of ngram models
         #e.g., dict of author: {1: (unigram model), 2: (bigram model), 3: (trigram model)}

    return models

def stupid_backoff(context: tuple[int, ...], ngram_len: int, next_token: int, models: dict[int, tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]]) -> float:
    #needs context, and all ngram models
    #use lambda = 0.4
    lambda_val = 0.4
    #go backwards from ngram count to unigram count as needed
    for ngram in range(ngram_len, 0, -1): 
        ngram_counts, context_counts = models[ngram]
        #build context, base case is 1
        if ngram == 1:
            ngram_key = (next_token,)
            #ctx so doesn't overwrite context variable
            ctx = () #empty
        else:
            #context was a tuple so neeed this to be a tuple, causes issues otherwise
            ctx = tuple(context[-(ngram-1):]) #ex: if 3gram, grabs last two tokens
            ngram_key = ctx + (next_token,)
        
        #if we have seen ngram, no issues! yay :) other backoff
        if ngram_key not in ngram_counts:
            #if not seen, backoff
            if ngram == 1:
                #if unigram and not seen, prob is zero. sad :(
                return 0.0
            else:
                #backoff m
                continue
        else:
            #we need to deal with if unigram vs. other
            if ngram == 1:
                probability = ngram_counts[ngram_key] / sum(ngram_counts.values()) #unigram prob
                #count(token)/total tokens
            else:
                probability = ngram_counts[ngram_key] / context_counts[ctx] #conditional prob
                #prob next word given context
    
            #apply backoff factor
            score = lambda_val ** (ngram_len - ngram) * probability
            return score
    
    #print
    print("exited loop, never found anything. is there an issue?")
    return 0.0


def calculate_perplexity(line: list[int], ngram_len: int, model: dict[int, tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]]) -> float:
    #do by line, return perplexity for that line only

    #call stupid backoff if freq = 0

    ####IF LINE IS SHORTER THAN NGRAM LEN### ###DO WE WANT TO NEED WITH THIS###
    if len(line) < ngram_len:
        print("Warning: line shorter than ngram len, investigate/deal with this. makes it not possible to find perplexity")

    #placeholders
    log_sum = 0.0
    predictions = 0

    ##find context and next token
    for i in range(len(line) - ngram_len + 1):
        #ngram, context, next tokens
        ngram = line[i:i+ngram_len] #grab line[0:ngram] so if ngram is 3, grabs 0,1,2
        context = ngram[0:-1]
        next_token = ngram[-1]

        #find probability of next token given context
        stupid_backoff_score = stupid_backoff(context, ngram_len, next_token, model)
        ###should probably rewrite this part, not the most elegant
        if stupid_backoff_score > 0:
            log_sum += math.log(stupid_backoff_score) #keep running log sum of probabilities
            predictions += 1
        else:
            #if this happens a lot, investigate further
            #print("Warning: zero probability encountered in perplexity calculation. Adding small nonzero probability.")
            stupid_backoff_score = 1e-20 #small value to avoid log(0)
            log_sum += math.log(stupid_backoff_score)
            predictions += 1

    #perplexity
    if predictions == 0:
        return float('inf') #if no predictions, return infinity
    average_logprob = log_sum / predictions
    perplexity = math.exp(-average_logprob)   
    return perplexity


def predict_author(line_token: list[int], ngram_len: int, models: dict[str, dict[int, tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]]]) -> str:
    #for each author, calculate perplexity of line, then predict author with lowest perplexity

    #initialize 
    lowest_perplexity = float('inf') #anything less than infinity
    author_pred = None

    for author, model in models.items():
        perplexity = calculate_perplexity(line_token, ngram_len, model)
        #if perplexity returns infinity, may need to investigate causes. could be an issue
        if perplexity == float('inf'):
            print(f"Warning: Author {author} model returned infinite perplexity for line. Investigate further if needed")
        
        if perplexity < lowest_perplexity:
            lowest_perplexity = perplexity
            #if lowest perplexity, update author_pred
            author_pred = author
    
    return author_pred
    #return author with lowest perplexity


def dev_test_results(ngram_len: int, models: dict[str, dict[int, tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]]], dev_set_dict: dict[str, list[str]]) -> None:
    #will have to tokenize the dev set too
    enc = tiktoken.get_encoding("o200k_base")

    for author, text in dev_set_dict.items():
        #tokenized version
        tokenized_text = tokenize_text(text, enc, ngram_len) #tokenize text is list[list[int]]

        #initialize so we can find accuracy %
        total_lines = len(tokenized_text)
        correct_lines = 0

        for line in tokenized_text:
            #for each line, predict author
            predicted_author = predict_author(line, ngram_len, models)
            #if matches actual author, is correct count
            if predicted_author == author:
                correct_lines += 1

        #then find percent correct
        percent_correct = (correct_lines / total_lines)*100 
        #print result with formatting specified
        print(f"{author:<10} {percent_correct:>5.2f}% correct")

        #we will classify author based on which gives lowest perplexity
        #so calculate perplexity based on text  
        #for each line, calculate perplexity for each author model
        #then average perplexities over all lines for that author
        #choose author with lowest perplexity
        #print out that author name and % correct

def test_file_results(ngram_len: int, models: dict[str, dict[int, tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]]], test_lines: list[str]) -> None:
    #ugh i hate the type hint for models, too long
    enc = tiktoken.get_encoding("o200k_base")

    #classify test lines, will need to tokenize them
    tokenized_test_lines = tokenize_text(test_lines, enc, ngram_len)
    #then iterate through the tokenized test lines
    for line in tokenized_test_lines:
        #predict author for each line, we can't do % correct here
        predicted_author = predict_author(line, ngram_len, models)
        #predicted author for each line
        print(predicted_author)
        

