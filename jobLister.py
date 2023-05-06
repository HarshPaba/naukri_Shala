# from pyresparser import ResumeParser
# import os
from docx import Document

# import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))
df =pd.read_csv('job_final.csv') 

df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))
df['test']


# import docx2txt
import nltk
 
nltk.download('stopwords')

# from pdf2docx import Converter
# import os

# # # dir_path for input reading and output files & a for loop # # #

from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

 
# you may read the database from a csv file or some other database
SKILLS = [
    'machine learning',
    'data science',
    'python',
    'word',
    'excel',
    'English',
    'ml',
    'WEb'
]
SKILLS_DB=[]
for i in SKILLS:
    SKILLS_DB.append(i.lower())
# SKILLS_DB=SKILLS_DB.lower()
# def extract_text_from_docx(docx_path):
#     txt = docx2txt.process(docx_path)
#     if txt:
#         return txt.replace('\t', ' ')
#     return None
 
# def main():
def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
 
    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)
 
    return found_skills
 
text=extract_text_from_pdf('resume.pdf')
skills = extract_skills(text)
resume=skills

skills=[]
skills.append(' '.join(word for word in resume))

import re

from ftfy import fix_text

def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

from sklearn.feature_extraction.text import TfidfVectorizer
# import re

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(skills)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
test = (df['test'].values.astype('U'))

def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices


distances, indices = getNearestN(test)
test = list(test) 
matches = []

for i,j in enumerate(indices):
    dist=round(distances[i][0],2) 
    temp = [dist]
    matches.append(temp)
    
matches = pd.DataFrame(matches, columns=['Match confidence'])

df['match']=matches['Match confidence']
df1=df.sort_values('match')
df1=df1[['Position','Company','Location','url']].head(10).reset_index()
df1=df1.drop('index',axis=1)
# df1.to_csv('op.csv')
# print(df1)