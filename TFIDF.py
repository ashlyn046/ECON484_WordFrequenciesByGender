import pandas as pd
import numpy as np
import json as json
import dask.bag as db
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer

# reading in the data
wordsDf = pd.read_csv('/content/gdrive/My Drive/RA/final.csv')

# Term Frequency-Inverse Document Frequency (TF-IDF) Vector to convert to frequency vectors
# show relative freq of diff terms by gender

#max features makes it so it can't generate more than 500 features. ngram_range(1,1) says we only look at unigrams
tfidf = TfidfVectorizer(max_features=500, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))

final_tfIdf = tfidf.fit_transform(wordsDf['processedtext'])
print(final_tfIdf)

df = pd.DataFrame(final_tfIdf.toarray(), columns = tfidf.get_feature_names())
print(df)

# tfidf = TfidfVectorizer(min_df=3)
tfidf.fit(list(subject_sentences.values()))
feature_names = tfidf.get_feature_names()


# Now we can write the transformation logic like this
def get_ifidf_for_words(text):
    tfidf_matrix= tfidf.transform([text]).todense()
    feature_index = tfidf_matrix[0,:].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    return dict(tfidf_scores)
