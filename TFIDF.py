import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Reading in the data
wordsDf = pd.read_csv('/content/gdrive/My Drive/RA/final.csv')

# Creating a CountVectorizer object, and generating word counts
cv=CountVectorizer() 
word_count_vector=cv.fit_transform(wordsDf)

# Creating a vectorizer object and fitting a vector
tfidf_vectorizer = TfidfVectorizer(max_features=500, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
tfidf_vector = tfidf_vectorizer.fit_transform(wordsDf['processedtext'])

# Create a dataframe from the rfidf vector (set feature names=words as columns, abstracts are rows)
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=wordsDf['processedtext'], columns=tfidf_vectorizer.get_feature_names_out())

# Reshaping the data so the words become the rows, and then renaming cols
tfidf_df = tfidf_df.stack().reset_index()
# print(tfidf_df)
tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'processedtext': 'doc','level_1': 'term'})
print(tfidf_df.head())

# Sorting the dataframe by document and tfidf (grouped by doc) and taking the first 10 vals to fetermine the top 5 important words from each abstract
top_ten_df = tfidf_df.sort_values(by=['doc','tfidf'], ascending=[True,False]).groupby(['doc']).head(5)
