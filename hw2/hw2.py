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

# read relevances
relevant_documents = []
for r in range(225):
    f = open("./r/" + str(r + 1) + ".txt")
    lines = f.readlines()
    relevant_documents.append([int(x) for x in lines])


# init vectorizers
binary_vectorizer = CountVectorizer(binary=True)
tf_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

# prepare matrices
matrices = {
    "binary": binary_vectorizer.fit_transform(corpus),
    "tf": tf_vectorizer.fit_transform(corpus),
    "tfidf": tfidf_vectorizer.fit_transform(corpus)
}

results = {
    "precision": {
        "binary": {
            "euclidean_distance": [],
            "cosine_similarity": []
        },
        "tf": {
            "euclidean_distance": [],
            "cosine_similarity": []
        },
        "tfidf": {
            "euclidean_distance": [],
            "cosine_similarity": []
        }
    },
    "recall": {
        "binary": {
            "euclidean_distance": [],
            "cosine_similarity": []
        },
        "tf": {
            "euclidean_distance": [],
            "cosine_similarity": []
        },
        "tfidf": {
            "euclidean_distance": [],
            "cosine_similarity": []
        }
    },
    "f-measure": {
        "binary": {
            "euclidean_distance": [],
            "cosine_similarity": []
        },
        "tf": {
            "euclidean_distance": [],
            "cosine_similarity": []
        },
        "tfidf": {
            "euclidean_distance": [],
            "cosine_similarity": []
        }
    }
}

# compute cosine similarity and euclidean distance between all queries and all docs for all weightings
for weighting in matrices:
    X = matrices[weighting][len(corpus_d):len(corpus)]
    Y = matrices[weighting][0:len(corpus_d)]
    cosine_similarity_array = cosine_similarity(X, Y)
    euclidean_distances_array = euclidean_distances(X, Y)
    np.savetxt('./o/' + weighting + '-cosine_similarity.csv', cosine_similarity_array, delimiter=',')
    np.savetxt('./o/' + weighting + '-euclidean_distances.csv', euclidean_distances_array, delimiter=',')

    euclidean_distances_sorted_indices = euclidean_distances_array.argsort()
    cosine_similarity_sorted_indices = cosine_similarity_array.argsort()

    for query, relevant_documents_list in enumerate(relevant_documents):
        N = len(relevant_documents_list)
        euclidean_distances_relevant_documents = euclidean_distances_sorted_indices[query, :N]+1
        cosine_similarity_relevant_documents = cosine_similarity_sorted_indices[query, -N:][::-1]+1

        # compute metrics for euclidean distance
        tp_euclidean_distance = np.intersect1d(euclidean_distances_relevant_documents, relevant_documents_list)
        precision_euclidean_distance = len(tp_euclidean_distance) / N
        recall_euclidean_distance = len(tp_euclidean_distance) / len(relevant_documents_list)
        f_measure_euclidean_distance = 0 if precision_euclidean_distance + recall_euclidean_distance == 0 else 2 * (precision_euclidean_distance * recall_euclidean_distance) / (precision_euclidean_distance + recall_euclidean_distance)

        results["precision"][weighting]["euclidean_distance"].append(precision_euclidean_distance)
        results["recall"][weighting]["euclidean_distance"].append(recall_euclidean_distance)
        results["f-measure"][weighting]["euclidean_distance"].append(f_measure_euclidean_distance)

        # compute metrics for cosine similarity
        tp_cosine_similarity = np.intersect1d(cosine_similarity_relevant_documents, relevant_documents_list)
        precision_cosine_similarity = len(tp_cosine_similarity) / N
        recall_cosine_similarity = len(tp_cosine_similarity) / len(relevant_documents_list)
        f_measure_cosine_similarity = 0 if precision_cosine_similarity + recall_cosine_similarity == 0 else 2 * (precision_cosine_similarity * recall_cosine_similarity) / (precision_cosine_similarity + recall_cosine_similarity)

        results["precision"][weighting]["cosine_similarity"].append(precision_cosine_similarity)
        results["recall"][weighting]["cosine_similarity"].append(recall_cosine_similarity)
        results["f-measure"][weighting]["cosine_similarity"].append(f_measure_cosine_similarity)

for metric in results:
    for weighting in results[metric]:
        for score in results[metric][weighting]:
            print(weighting + ("\t\t" if weighting == "tf" else "\t")
                  + score + "\t"
                  + metric + ("\t\t" if metric == "recall" else "\t")
                  + "avg: " + str(sum(results[metric][weighting][score])/len(results[metric][weighting][score])))
