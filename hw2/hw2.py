# import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# prepare corpus
corpus_d = []
corpus_q = []
corpus = []
for d in range(1400):
    f = open("./d/" + str(d + 1) + ".txt")
    text = f.read()
    corpus_d.append(text)
    corpus.append(text)

# add query to corpus
for q in range(225):
    f = open("./q/" + str(q + 1) + ".txt")
    text = f.read()
    corpus_q.append(text)
    corpus.append(text)

# init vectorizers
binary_vectorizer = CountVectorizer(binary=True)
tf_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

# prepare matrices
matrices = {
    "binary_matrix": binary_vectorizer.fit_transform(corpus),
    "tf_matrix": tf_vectorizer.fit_transform(corpus),
    "tfidf_matrix": tfidf_vectorizer.fit_transform(corpus)
}

# compute cosine similarity and euclidian distance between all queries and all docs for all weightings
for index in matrices:
    print(index)
    X = matrices[index][len(corpus_d):len(corpus) - 1]
    Y = matrices[index][0:(len(corpus_d) - 1)]
    cosine_similarity_array = cosine_similarity(X, Y)
    euclidean_distances_array = euclidean_distances(X, Y)
    np.savetxt('./o/' + index + '-cosine_similarity.csv', cosine_similarity_array, delimiter=',')
    np.savetxt('./o/' + index + '-euclidean_distances.csv', euclidean_distances_array, delimiter=',')

    # topRelevant = cosine_similarity_array.argsort()[-10:][::-1] + 1
    # print(topRelevant)
