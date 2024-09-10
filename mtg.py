import nltk
import random
nltk.download('gutenberg')
nltk.download('punkt')


def train_n_gram(corpus, n):
    '''Given a corpus and a desired integer for the ngram, return count of n+1 token combinations'''
    context_dict = dict()
    for i in range(len(corpus) - n):
        token_comb = tuple(corpus[i:i+n])
        if token_comb[:-1] in context_dict:
            if token_comb[-1] in context_dict[token_comb[:-1]]:
                context_dict[token_comb[:-1]][token_comb[-1]] += 1
            else:
                context_dict[token_comb[:-1]][token_comb[-1]] = 1
        else:
            context_dict[token_comb[:-1]] = {token_comb[-1]: 1}  
    return context_dict

def up_to_n_gram_train(corpus, n):
    '''dict_list[unigram_dict, bigram_dict..., ngram_dict]'''
    dict_list = []
    # create n-gram distribiution for i in range(1,n)
    for i in range(n):
        dict_list.append(train_n_gram(corpus, i+1))
    return dict_list

def S(word, context, dict_list, alpha = 0.4):
    ''' Find the score of a specific word given a context and the dictionaries wrt the corpus'''
    n = len(context)
    context_dict = dict_list[n]
    if context in context_dict:
        if word in context_dict[context]:
            return context_dict[context][word]/sum(context_dict[context].values())
        else:
            return alpha * S(word, context[1:], dict_list)
    else:
        return alpha * S(word, context[1:], dict_list)

def predict_next(context, dict_list, randomize=False):
    '''Given the context, the next token will be predicted'''
    S_vocab_context = dict()
    for word in dict_list[0][tuple()]:
        S_vocab_context[word] = S(word, context, dict_list)
    if not randomize:
        sorted_next = sorted(S_vocab_context.items(), key = lambda x:(-x[1],x[0]), reverse = False)
        return sorted_next[0][0]
    else:
        words = list(S_vocab_context.keys())
        weights = list(S_vocab_context.values())
        return random.choices(words, weights)[0]
        
def finish_sentence(sentence, n, corpus, randomize=False):
    '''Predicts the tokens to follow the existing tokens in the sentence until 10 tokens or punctuation mark'''
    dict_list = up_to_n_gram_train(corpus, n)
    while sentence[-1] not in ['.','!','?'] and len(sentence) < 10:
        if len(sentence) < n-1:
            context = tuple(['']*(n - 1 - len(sentence)) + sentence)
        else:
            context = tuple(sentence[-n+1:])
        sentence.append(predict_next(context, dict_list, randomize))
    return sentence