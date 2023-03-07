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

# ngram_range(1,1) only looking at unigrams initially
tfidf = TfidfVectorizer(max_features=500, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))

final_tfIdf = tfidf.fit_transform(wordsDf['processedtext'])
print(final_tfIdf)
