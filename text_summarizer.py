import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def read_article(file_name):
    file = open(file_name,'r')
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences  = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split())
    sentences.pop()

    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1  = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1  + sent2))

    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)

    #build the vector for the first sentence 
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1- cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    #Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_sumary(file_name, top_n = 5): 
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    sumarize_text = []

    #step1 - read text and split it
    sentences = read_article(file_name)

    #step2 - Generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    #step3 - rank sentences in similarity matrix
    sentences_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentences_similarity_graph)

    #step4 - sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    print('Indexes of top ranked_sentence order are:', ranked_sentence)

    for i in range(top_n):
        sumarize_text.append(" ".join(ranked_sentence[i][1]))

    #step5 - output the sumarize text
    print("Sumarize Text: \n",". ".join(sumarize_text))
    print('--------------------------------------------------------')
    out_put = open('sumarize_text','w')
    out_put.write(". ".join(sumarize_text))


generate_sumary('test.txt', 2)