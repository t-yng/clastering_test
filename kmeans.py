import scipy as sp
import sklearn.datasets
from classter import StemmedTfidfVectorizer
from sklearn.cluster import KMeans

MLCOMP_DIR = r"/Users/tomohiro/workspace/python/MachineLearningSystem/ch03/data/news"
groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
          'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR, categories=groups)

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)

num_clusters = 50
km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
km.fit(vectorized)

new_post = "Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks."
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

similar_indices = (km.labels_ == new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))
similar = sorted(similar)

print(similar[0][1].decode('utf-8'))

# for line in show_at_1[1].split("\n"):
#     print(line)

