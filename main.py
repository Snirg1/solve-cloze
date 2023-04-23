import json
import random

from nltk import word_tokenize
import os
import datetime

from numpy import mean

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + '/'


def generate_ngrams(tokens, k):
    l = []
    i = 0
    while (i < len(tokens)):
        l.append(tokens[i:i + k])
        i = i + 1
    l = l[:-1]
    return l


def preprocess(d):
    d = d.lower()
    d = d.replace(".", " eos")
    return d


def generate_tokens(d):
    tokens = word_tokenize(d)
    return tokens


def generate_tokens_freq(tokens):
    dct = {}
    for i in tokens:
        dct[i] = 0
    for i in tokens:
        dct[i] += 1

    return dct


def generate_ngram_freq(bigram):
    dct = {}
    for i in bigram:
        st = " ".join(i)
        dct[st] = 0
    for i in bigram:
        st = " ".join(i)
        dct[st] += 1
    return dct


def find_val_in_dict(s, dct1):
    try:
        return dct1[s]
    except:
        return 0


def get_pre_words(cloze_tokens):
    preWords = []
    n = len(cloze_tokens)
    for i in range(n):
        if cloze_tokens[i] == '__________':
            preWord = cloze_tokens[i - 1]
            preWords.append(preWord)
    return preWords


def get_post_words(cloze_tokens):
    postWords = []
    n = len(cloze_tokens)
    for i in range(n):
        if cloze_tokens[i] == '__________':
            postWord = cloze_tokens[i + 1]
            postWords.append(postWord)
    return postWords


def calc_prob_of(distinct_tokens, dct, dct1, first, second):
    denominator = dct[distinct_tokens[first]]
    numerator = find_val_in_dict(distinct_tokens[first] + " " + distinct_tokens[second], dct1)
    prob = float("{:.3f}".format(numerator / denominator))
    return prob


def check_combination(candidates, words, distinct_tokens, token_freq_dict, ngram_freq_dict, order='preWords'):
    firstWordIndex = 0
    secondWordIndex = 0
    text = ""
    dict = {}
    for word in words:
        for candidate in candidates:
            try:
                if order == 'preWords':
                    text = word + " " + candidate
                    firstWordIndex = distinct_tokens.index(word)
                    secondWordIndex = distinct_tokens.index(candidate)
                if order == 'postWords':
                    text = candidate + " " + word
                    firstWordIndex = distinct_tokens.index(candidate)
                    secondWordIndex = distinct_tokens.index(word)
                text = preprocess(text)
                prob = calc_prob_of(distinct_tokens, token_freq_dict, ngram_freq_dict, firstWordIndex, secondWordIndex)
                if prob > 0:
                    dict[("").join(text)] = prob
            except:
                pass
    return dict


def get_blank_index(cloze_tokens, word, order="pre"):
    indices = [i for i, x in enumerate(cloze_tokens) if x == word]
    res = 0

    try:
        for ind in indices:
            if order == "pre":
                if cloze_tokens[ind + 1] == '__________':
                    res = ind + 1
                    return res
            if order == "post":
                if ind > 0:
                    if cloze_tokens[ind - 1] == '__________':
                        res = ind - 1
                        return res
    except:
        pass


def tokenize_corpus(f_corpus):
    tokens = []
    now = datetime.datetime.now()
    time_limit = datetime.timedelta(minutes=19) + now

    # read lines from corpus file as long our runtime is under 20 minutes
    while True:
        # make sure that our runtime is maximum of 20 minutes
        now = datetime.datetime.now()
        if now > time_limit:
            return tokens
        # read single line from text file
        line = f_corpus.read(1024)
        if not line:
            break
        # tokenize the line and add it to the tokens list
        tokens = tokens + generate_tokens(preprocess(line))
    return tokens


def get_blank_indices(cloze_tokens, pre_words, post_words, dict_of_probabilities_pre, dict_of_probabilities_post):
    blank_indices = []

    for w in pre_words:
        i = get_blank_index(cloze_tokens, w, 'pre')
        blank_indices.append(i)
        dict_of_probabilities_pre[i] = ("", 0)

    for w in post_words:
        i = get_blank_index(cloze_tokens, w, 'post')
        blank_indices.append(i)
        dict_of_probabilities_post[i] = ('', 0)

    return blank_indices


def fill_random_cand(lst, i, candidates):
    for cand in candidates:
        if cand not in lst:
            lst[i] = cand
            break


def fill_empty_with_random_candidates(lst, candidates):
    n = len(lst)
    for i in range(n):
        if lst[i] == '':
            fill_random_cand(lst, i, candidates)


def unite_pre_post_dicts(cloze_tokens, dict_of_probabilities_pre, dict_of_probabilities_post, dict_pre,
                         dict_post):
    for key, value in dict_pre.items():
        if value:
            word = (key.split(" "))[0]
            cand = (key.split(" "))[1]
            blank_index = get_blank_index(cloze_tokens, word, 'pre')
            if value > dict_of_probabilities_pre[blank_index][1]:
                dict_of_probabilities_pre[blank_index] = (cand, value)

    for key, value in dict_post.items():
        if value:
            word = (key.split(" "))[1]
            cand = (key.split(" "))[0]
            blank_index = get_blank_index(cloze_tokens, word, 'post')
            if value > dict_of_probabilities_post[blank_index][1]:
                dict_of_probabilities_post[blank_index] = (cand, value)


def prepare_unique_prob_dict(dict_of_probabilities_pre, dict_of_probabilities_post, blank_indices):
    dictIndFinal = {}
    for i in blank_indices:
        dictIndFinal[i] = [("", 0.0)]

    for indPre, tPre in dict_of_probabilities_pre.items():
        if tPre[1] != 0:
            dictIndFinal[indPre].append(tPre)

    for indPost, tPost in dict_of_probabilities_post.items():
        if tPost[1] != 0:
            dictIndFinal[indPost].append(tPost)

    for ind, list_of_tuples in dictIndFinal.items():
        dictIndFinal[ind] = sorted(list_of_tuples, key=lambda t: t[1], reverse=True)
    return dictIndFinal


def build_word_max_prob_dict(dict_of_probabilities, candidates):
    maxDict = {}
    for c in candidates:
        if c != '':
            maxDict[c] = 0.0

    for ind, list_of_tuples in dict_of_probabilities.items():
        for t in list_of_tuples:
            if t[0] != '':
                if maxDict[t[0]] < t[1]:
                    maxDict[t[0]] = t[1]

    return maxDict


def filter_not_maximum_values(dict_of_probabilities, max_dict):
    for ind, list_of_tuples in dict_of_probabilities.items():
        try:
            if dict_of_probabilities[ind][0][1] != max_dict[dict_of_probabilities[ind][0][0]]:
                dict_of_probabilities[ind].remove(dict_of_probabilities[ind][0])
        except:
            pass


def build_final_list_from_final_dict(dict_of_probabilities):
    lst = []
    for ind, list_of_tuples in dict_of_probabilities.items():
        tup = list_of_tuples[0]
        if tup[0] == '':
            lst.append(tup[0])
        else:
            if tup[0] not in lst:
                lst.append(tup[0])
    return lst


def get_candidate_from_file(candidates):
    with open(dir_path + candidates) as fCandidates:
        candidates = fCandidates.read().split("\n")
        n = len(candidates)
        if candidates[n - 1] == '':
            candidates.pop()
    return candidates


def get_cloze_tokens_from_file(input):
    with open(dir_path + input) as fCloze:
        cloze_tokens = word_tokenize(preprocess(fCloze.read()))
    return cloze_tokens


def build_dict_of_probs(candidates, cloze_tokens, pre_words, post_words, distinct_tokens, token_freq_dict,
                        ngram_freq_dict):
    dict_of_probabilities_pre = {}
    dict_of_probabilities_post = {}
    # initialize 2 dictionaries of pre- / post-words and calculate all the blank indices from the cloze

    blank_indices = get_blank_indices(cloze_tokens, pre_words, post_words, dict_of_probabilities_pre,
                                      dict_of_probabilities_post)
    blank_indices = sorted(list(dict.fromkeys(blank_indices)))

    # check probabilities of different combinations that includes: candidates words & pre- or post-words from cloze
    unfiltered_dict_of_probabilities_pre = check_combination(candidates, pre_words, distinct_tokens, token_freq_dict,
                                                             ngram_freq_dict, "preWords")

    unfiltered_dict_of_probabilities_post = check_combination(candidates, post_words, distinct_tokens,
                                                              token_freq_dict, ngram_freq_dict, "postWords")

    unite_pre_post_dicts(cloze_tokens, dict_of_probabilities_pre, dict_of_probabilities_post,
                         unfiltered_dict_of_probabilities_pre, unfiltered_dict_of_probabilities_post)

    dict_of_probabilities = prepare_unique_prob_dict(dict_of_probabilities_pre, dict_of_probabilities_post,
                                                     blank_indices)

    return dict_of_probabilities


def calc_random_mean_accuracy():
    rand = random.SystemRandom()
    correct_solution = ['times', 'bridge', 'concrete']
    rand_solution = ['times', 'bridge', 'concrete']
    list_of_rand_solutions_accuracy = []

    for i in range(100):
        # init test correct_solution and correct answers counter
        correct_answers_counter = 0
        rand.shuffle(rand_solution)
        # for every correct guess, add 1 to correct answers counter
        if rand_solution[0] == correct_solution[0]:
            correct_answers_counter += 1
        if rand_solution[1] == correct_solution[1]:
            correct_answers_counter += 1
        if rand_solution[2] == correct_solution[2]:
            correct_answers_counter += 1
        # calculate the current correct_solution accuracy
        accuracy = correct_answers_counter / 3
        # add the current correct_solution accuracy to the list
        list_of_rand_solutions_accuracy.append(accuracy)

    # calculate the mean accuracy of 100 random guesses
    mean_of_accuracy = mean(list_of_rand_solutions_accuracy)
    return mean_of_accuracy


def solve_cloze(input, candidates, lexicon, corpus):
    print(f'starting to solve the cloze {input} with {candidates} using {lexicon} and {corpus}')

    # extract list of candidates from file of candidates
    candidates = get_candidate_from_file(candidates)

    # extract list of cloze tokens from input cloze file
    cloze_tokens = get_cloze_tokens_from_file(input)

    # get all the words that come before/after ' _________ ' in the cloze
    preWords = get_pre_words(cloze_tokens)
    postWords = get_post_words(cloze_tokens)

    # tokenize corpus text file
    with open(dir_path + corpus) as f_corpus:
        tokens = tokenize_corpus(f_corpus)

    # create unique tokens list
    distinct_tokens = list(set(sorted(tokens)))

    # generate bi-grams from tokens
    bigrams = generate_ngrams(tokens, 2)

    # build dictionary of probs - this one contains duplications and not max values only
    dict_of_probs = build_dict_of_probs(candidates, cloze_tokens, preWords, postWords, distinct_tokens,
                                        generate_tokens_freq(tokens),
                                        generate_ngram_freq(bigrams))

    # build a dictionary of {word: max_prob} so we can make sure a word will be in the highest probability spot for it
    max_prob_dict = build_word_max_prob_dict(dict_of_probs, candidates)
    # filter all the candidates with probability that is under the max probability
    filter_not_maximum_values(dict_of_probs, max_prob_dict)
    # now our dict of probabilities is ready to be translated to list as required
    lst = build_final_list_from_final_dict(dict_of_probs)
    # fill empty left blanks by replace it with random unused candidate from candidates list
    fill_empty_with_random_candidates(lst, candidates)

    return lst  # return your solution


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    # calc the mean of accuracy of 100 random solutions
    chance_accuracy = calc_random_mean_accuracy()
    # print("chance accuracy is: %.3f", chance_accuracy)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['lexicon_filename'],
                           config['corpus'])
    print('cloze solution:', solution)
