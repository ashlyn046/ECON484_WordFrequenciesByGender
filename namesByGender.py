# Dataset of names by gender from https://archive.ics.uci.edu/ml/datasets/Gender+by+Name

namesByGender = pd.read_csv('/content/gdrive/My Drive/RA/name_gender_dataset.csv')
namesByGender.rename(columns = {'Name':'firstName'}, inplace = True)

# All of the names appear two times, one as female and one as male with probabilities. Here, I drop the ones with a lower probability
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
smallerNames.to_csv('/content/gdrive/My Drive/RA/cleanedNames.csv') 
