import os
import sys
import scipy as sp
from nltk import stem
from sklearn.feature_extraction.text import TfidfVectorizer

english_stemmer = stem.SnowballStemmer('english')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

# DIR = r"data/txt/"
#
# posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
#
# vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')
# X_train = vectorizer.fit_transform(posts)
#
# samples, features = X_train.shape
# print(samples, features)
#
# new_post = "imaging databases"
# new_post_vec = vectorizer.transform([new_post])
#
# best_doc = None
# best_dist = sys.maxsize
# best_i = None
# for i in range(0, samples):
#     post = posts[i]
#     if post == new_post:
#         continue
#     post_vec = X_train.getrow(i)
#     d = dist_norm(post_vec, new_post_vec)
#     print("=== Post %i with dist=%.2f: %s" % (i, d, post))
#     if d < best_dist:
#         best_dist = d
#         best_i = i
# print("Best post is %i with dist=%.2f" % (best_i, best_dist))
