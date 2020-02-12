import nltk
import sys
import re
import math
from nltk.tokenize import word_tokenize


def clean(corpus):
    cleaned_corpus = corpus.lower()
    cleaned_corpus = re.sub(' +', ' ', cleaned_corpus)
    cleaned_corpus = re.sub('-+', ' ', cleaned_corpus)
    cleaned_corpus = re.sub('[^a-zA-Z \n]+', '', cleaned_corpus)

    return cleaned_corpus


def sentencizer(text):
    temp = text.split("\n")

    sentences = []

    for s in temp:
        if s.strip():
            sentences.append(s.strip())
            # sentences.append("$ " + s.strip() + " #")

    return sentences


def get_unigrams(sentences):
    unigrams = {}

    for sentence in sentences:
        for word in sentence:

            if word not in unigrams:
                unigrams[word] = 1

            else:
                unigrams[word] += 1

    return unigrams


def get_bigrams(sentences):
    bigrams = {}

    for sentence in sentences:
        length = len(sentence)

        for i in range(length - 1):
            if sentence[i] not in bigrams:
                bigrams[sentence[i]] = {}

            if sentence[i+1] not in bigrams[sentence[i]]:
                bigrams[sentence[i]][sentence[i+1]] = 1

            else:
                bigrams[sentence[i]][sentence[i+1]] += 1

    return bigrams


def get_trigrams(sentences):
    trigrams = {}

    for sentence in sentences:
        length = len(sentence)

        for i in range(length - 2):
            if sentence[i] not in trigrams:
                trigrams[sentence[i]] = {}

            if sentence[i+1] not in trigrams[sentence[i]]:
                trigrams[sentence[i]][sentence[i+1]] = {}

            if sentence[i+2] not in trigrams[sentence[i]][sentence[i+1]]:
                trigrams[sentence[i]][sentence[i+1]][sentence[i+2]] = 1

            else:
                trigrams[sentence[i]][sentence[i+1]][sentence[i+2]] += 1

    return trigrams


def unigram_vocabulary(unigrams):
    return len(unigrams)


def total_unigrams(unigrams):
    cnt = 0

    for i in unigrams:
        cnt += unigrams[i]

    return cnt


def bigram_vocabulary(bigrams):
    cnt = 0

    for i in bigrams:
        cnt += len(bigrams[i])

    return cnt


def trigram_vocabulary(trigrams):
    cnt = 0

    for i in trigrams:
        for j in trigrams[i]:
            cnt += len(trigrams[i][j])

    return cnt


def kneser_ney_unigrams(unigrams, tokenized_inp):
    cur = 0
    d = 0.25

    total = total_unigrams(unigrams)

    for sentence in tokenized_inp:
        length = len(sentence)

        for i in range(length):

            cnt = 0

            if sentence[i] in unigrams:
                cnt = unigrams[sentence[i]]

            # vocabulary = unigram_vocabulary(unigrams)
            # print(cnt)
            prob = max(cnt - d, 0)/total
            # print(prob)
            prob += (d/total)

            cur += math.log(prob)

    return math.exp(cur)


def kneser_ney_bigrams(bigrams, tokenized_inp, unigrams):
    cur = 0
    d = 0.75

    vocabulary = bigram_vocabulary(bigrams)
    uni_vocabulary = unigram_vocabulary(unigrams)
    tot_unigrams = total_unigrams(unigrams)

    for sentence in tokenized_inp:
        length = len(sentence)

        for i in range(1, length):

            if sentence[i - 1] not in unigrams:
                p = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(p)

            else:
                cnt = 0

                if sentence[i] in bigrams[sentence[i-1]]:
                    cnt = max(bigrams[sentence[i - 1]][sentence[i]] - d, 0)

                den = unigrams[sentence[i - 1]]

                p = cnt/den

                lambda1 = ((d/unigrams[sentence[i-1]])*len(bigrams[sentence[i-1]]))

                tot = 0

                for j in bigrams:
                    if sentence[i] in bigrams[j]:
                        tot += 1

                p2 = max(tot - d, 0)/vocabulary
                lambda2 = (d/tot_unigrams)*uni_vocabulary
                p2 += (lambda2/uni_vocabulary)

                p += (lambda1 * p2)

                cur += math.log(p)

    return math.exp(cur)


def kneser_ney_trigrams(trigrams, tokenized_inp, unigrams, bigrams):
    d = 9
    cur = 0

    vocabulary = trigram_vocabulary(trigrams)
    bi_vocabulary = bigram_vocabulary(bigrams)
    uni_vocabulary = unigram_vocabulary(unigrams)
    tot_unigrams = total_unigrams(unigrams)

    for sentence in tokenized_inp:
        length = len(sentence)

        for i in range(2, length):
            if sentence[i-2] not in unigrams:
                prob = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(prob)

            elif sentence[i-1] not in unigrams:
                prob = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(prob)

            elif sentence[i-1] not in bigrams[sentence[i-2]]:
                prob = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(prob)

            else:
                cnt = 0

                if sentence[i] in trigrams[sentence[i-2]][sentence[i-1]]:
                    cnt = max(trigrams[sentence[i-2]][sentence[i-1]][sentence[i]] - d, 0)

                den = bigrams[sentence[i-2]][sentence[i-1]]

                p = cnt/den

                lambda1 = ((d/bigrams[sentence[i - 2]][sentence[i - 1]]) *
                           len(trigrams[sentence[i - 2]][sentence[i - 1]]))
                tot1 = 0
                tot1_den = 0

                for j in trigrams:
                    if sentence[i - 1] in trigrams[j]:
                        if sentence[i] in trigrams[j][sentence[i - 1]]:
                            tot1 += 1

                        tot1_den += len(trigrams[j][sentence[i - 1]])

                p2 = max(tot1 - d, 0)/tot1_den

                lambda2 = ((d/unigrams[sentence[i - 1]]) * len(bigrams[sentence[i - 1]]))

                tot2 = 0

                for j in bigrams:
                    if sentence[i] in bigrams[j]:
                        tot2 += 1

                p3 = max(tot2 - d, 0)/bi_vocabulary
                lambda3 = ((d/tot_unigrams)*uni_vocabulary)
                p3 += lambda3/uni_vocabulary

                p2 += lambda2*p3
                p += lambda1*p2

                cur += math.log(p)

    return math.exp(cur)


def witten_bell_unigrams(unigrams, tokenized_inp):
    cur = 0
    d = 0.25

    total = total_unigrams(unigrams)
    # print(total)

    for sentence in tokenized_inp:
        for word in sentence:
            if word not in unigrams:
                cur += math.log((d/total))
                # print(d/total)

            else:
                cnt = unigrams[word]
                p = cnt / (total + len(unigrams))
                cur += math.log(p)
                # print(p)

    return math.exp(cur)


def witten_bell_bigrams(bigrams, tokenized_inp, unigrams):
    cur = 0
    d = 0.75

    vocabulary = bigram_vocabulary(bigrams)
    uni_vocabulary = unigram_vocabulary(unigrams)
    tot_unigrams = total_unigrams(unigrams)

    for sentence in tokenized_inp:
        length = len(sentence)

        for i in range(1, length):

            if sentence[i - 1] not in unigrams:
                p = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(p)

            elif sentence[i] not in unigrams:
                p = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(p)

            else:
                tot = len(bigrams[sentence[i - 1]])

                add_den = tot + unigrams[sentence[i - 1]]

                lambda1 = 1 - tot/add_den

                if sentence[i] in bigrams[sentence[i - 1]]:
                    first_term = bigrams[sentence[i - 1]][sentence[i]]

                else:
                    first_term = (len(bigrams[sentence[i - 1]]) /
                                  (uni_vocabulary - len(bigrams[sentence[i - 1]])))

                first_term_den = unigrams[sentence[i - 1]]

                first_term_den += len(bigrams[sentence[i - 1]])
                first_term /= first_term_den

                second_term = witten_bell_unigrams(unigrams, [nltk.word_tokenize(sentence[i])])

                p = lambda1 * first_term + (1 - lambda1) * second_term

                # print(lambda1, first_term, second_term, p)

                cur += math.log(p)

    return math.exp(cur)


def witten_bell_trigrams(trigrams, tokenized_inp, unigrams, bigrams):
    cur = 0
    d = 9

    tot_unigrams = total_unigrams(unigrams)
    uni_vocabulary = unigram_vocabulary(unigrams)
    vocabulary = trigram_vocabulary(trigrams)

    for sentence in tokenized_inp:
        length = len(sentence)

        for i in range(2, length):

            if sentence[i - 2] not in unigrams:
                prob = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(prob)

            elif sentence[i - 1] not in unigrams:
                prob = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(prob)

            elif sentence[i - 1] not in bigrams[sentence[i - 2]]:
                prob = (d/tot_unigrams)*(uni_vocabulary/vocabulary)
                cur += math.log(prob)

            else:
                tot = len(trigrams[sentence[i - 2]][sentence[i - 1]])

                tot_den = tot + bigrams[sentence[i - 2]][sentence[i - 1]]

                lambda1 = 1 - (tot/tot_den)

                if sentence[i] in trigrams[sentence[i - 2]][sentence[i - 1]]:
                    first_term = trigrams[sentence[i - 2]][sentence[i - 1]][sentence[i]]

                else:
                    first_term = len(trigrams[sentence[i - 2]][sentence[i - 1]]) / \
                        (uni_vocabulary - len(trigrams[sentence[i - 2]][sentence[i - 1]]))

                first_term_den = len(trigrams[sentence[i - 2]][sentence[i - 1]]
                                     ) + bigrams[sentence[i - 2]][sentence[i - 1]]
                first_term /= first_term_den

                second_term = witten_bell_bigrams(
                    bigrams, [nltk.word_tokenize(sentence[i - 1] + " " + sentence[i])], unigrams)

                # print([nltk.word_tokenize(sentence[i - 1] + " " + sentence[i])])

                p = lambda1 * first_term + (1 - lambda1) * second_term
                # print(p)
                cur += math.log(p)

    return math.exp(cur)


n = sys.argv[1]
type = sys.argv[2]
path = sys.argv[3]

file = open(path)
corpus = file.read()
cleaned_corpus = clean(corpus)
sentences = sentencizer(cleaned_corpus)
sentences = [nltk.word_tokenize(s) for s in sentences]

unigrams = get_unigrams(sentences)
bigrams = get_bigrams(sentences)
trigrams = get_trigrams(sentences)

# print(unigrams)
# print()
# print(bigrams)
# print()
# print(trigrams)

print("input sentence:", end=" ")
sent = input()

sent = clean(sent)
sent = sentencizer(sent)
sent = [nltk.word_tokenize(s) for s in sent]

# print(sent)
if sent:

    if n == "1" and type == "k":
        print(kneser_ney_unigrams(unigrams, sent))

    elif n == "2" and type == "k":
        print(kneser_ney_bigrams(bigrams, sent, unigrams))

    elif n == "3" and type == "k":
        print(kneser_ney_trigrams(trigrams, sent, unigrams, bigrams))

    elif n == "1" and type == "w":
        print(witten_bell_unigrams(unigrams, sent))

    elif n == "2" and type == "w":
        print(witten_bell_bigrams(bigrams, sent, unigrams))

    elif n == "3" and type == "w":
        print(witten_bell_trigrams(trigrams, sent, unigrams, bigrams))
