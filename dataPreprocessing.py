import pandas as pd
import numpy as np
import json as json
import dask.bag as db
import nltk
nltk.download('punkt')

# STEP 1: Reading in the papers
papers = pd.read_csv('/content/gdrive/My Drive/RA/papers.csv')
authors = pd.read_csv('/content/gdrive/My Drive/RA/authors.csv')

#merging
merged = pd.merge(papers, authors, on="id")

# dropping rows where abstract is missing
merged = merged[merged.abstract != "Abstract Missing"]

# keeping only very relevant columns
cols = ['name', 'abstract']
merged = merged[cols]

#creating a first name column for later merge
merged["firstName"] = merged["name"].str.partition(' ')[0]

# Preprocessing the words
nltk.download('stopwords')
STOP_WORDS = nltk.corpus.stopwords.words()

# preprocessing text? 
merged['processedtext'] = merged['abstract'].str.replace('[^\w\s]','') #takes out special characters

#this takes out the stop words
merged['processedtext'] = merged['processedtext'].apply(lambda x: " ".join(x for x in x.split() if x not in STOP_WORDS))

#this conversts everything to lower. x.split() lets you go char by char
merged['processedtext'] = merged['processedtext'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# stemming all the words (ex changes arguing, argues, argued into argu)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
merged['processedtext'] = merged['processedtext'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
merged = merged.drop_duplicates()


# STEP 2: Merging with the names by gender data
namesByGender = pd.read_csv('/content/gdrive/My Drive/RA/name_gender_dataset.csv')
namesByGender.rename(columns = {'Name':'firstName'}, inplace = True)

#All of the names appear two times, one as female and one as male with probabilities. Here, I drop the ones with a lower probability
smallerNames = pd.DataFrame()

for name in namesByGender.firstName:
    currnames = namesByGender.loc[namesByGender.firstName == name]
    currnames = currnames.reset_index(drop=True)
    if currnames.shape[0] ==1:
      pass
    elif currnames.iloc[0,:].Probability > currnames.iloc[1,:].Probability:
      currnames = currnames.drop(index=1)
    elif currnames.iloc[0,:].Probability <= currnames.iloc[1,:].Probability:
      currnames = currnames.drop(index=0)
    smallerNames = pd.concat([smallerNames, currnames], ignore_index=True)

smallerNames = smallerNames.drop_duplicates()
final = pd.merge(merged, smallerNames, on="firstName")

# STEP 3: Saving final file 
final.to_csv('/content/gdrive/My Drive/RA/final.csv')
