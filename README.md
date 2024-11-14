# WordFrequenciesByGender_Abstracts

This research project examines word frequency variations in academic articles written by authors of different genders. Specifically, it aims to analyze the linguistic differences between articles co-authored by men and women versus those authored solely by men, providing insights into potential gender-based disparities in academic language.

This was a collaborative effort. 

## Project Structure

The repository currently contains three main Python files:
- `namesByGender.py`
- `dataPreprocessing.py`
- `TFIDF.py`

### File Descriptions

#### 1. `namesByGender.py`

This script processes a name dataset to classify commonly used names by their likely gender.

- **Purpose**: Creates a cleaned dataset, `cleanedNames.csv`, by retaining the most probable gender for each name. This file is then used to predict the likely genders of authors in the abstracts dataset.
- **Data Source**: [UCI Gender by Name Dataset](https://archive.ics.uci.edu/ml/datasets/Gender+by+Name)
- **Method**: The script removes duplicate name entries with lower gender probability using a simple loop, resulting in a streamlined dataset.

#### 2. `dataPreprocessing.py`

This script preprocesses author and abstract datasets for analysis.

- **Purpose**: Merges two datasets (authors and abstracts) using an ID variable, cleans and formats the data, generates a `firstname` feature, and prepares text for analysis.
- **Steps**:
  - Merge datasets on ID variable.
  - Clean text data by removing special characters, stopwords, converting text to lowercase, and applying stemming.
  - Merge with the `cleanedNames.csv` file and output the preprocessed data as `final.csv`.

#### 3. `TFIDF.py`

**Status**: **In Progress**

- **Purpose**: Generates word frequency data to identify gender-based language patterns.
- **Method**: Utilizes Term Frequency-Inverse Document Frequency (TF-IDF) to determine the significance of words within each document in the dataset. This allows for ranking words by their relative importance, which may reveal gender-based differences in writing styles.

---

**Note**: More files and functionalities will be added as the project progresses.
