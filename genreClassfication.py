
# coding: utf-8

# ## Genre Classification

# In[ ]:


import re
import gensim
import nltk
import spacy
import math
import langid
import numpy as np
import pandas as pd
import en_core_web_sm
import string as string
from pprint import pprint
nlp = en_core_web_sm.load()
#nltk.download('stopwords')
from langdetect import detect
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.stem import PorterStemmer
# from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# https://github.com/saffsd/langid.py


# In[ ]:


songs = pd.read_csv(r"C:\Users\e9900331\Documents\IUB\Fall 2018\L 645\Final Project\lyrics.csv")


# In[ ]:


print(songs.shape)


# In[ ]:


#creating vector with stopword
stop_words = set(stopwords.words('english'))
print(stop_words)
# for word in stop_words:
#     print(word)


# In[ ]:


def replace_stopwords(text):
    words = nltk.word_tokenize(str(text))
    song = [word for word in words if(word not in stop_words)]
    return (" ".join(song))

# Ref: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

exclude = set(string.punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))

def remove_punct(s): 
    return regex.sub('', str(s))

# def remove_short_songs(s):
#     if(len(str(s)) > 20): return s
    
# Ref: https://stackoverflow.com/questions/26264579/select-the-rows-that-satisfy-the-same-condition-on-all-columns
def select(list_of_rows, filter_function):
    result = []
    for row in list_of_rows:
        if filter_function(row):
            result.append(row)
    return result

def str_filter(s):
    if (isinstance(s, str)):
        return True
    return False

def not_short(s):
    if (len(str(s)) > 150):
        return True
    return False

def is_English(s):
    if(detect(s) == 'en'):
        return True
    return False

# def get_alpha(s):
#     if(s.isalpha()):
#         return True
#     return False


# In[ ]:


#clean data
songs = songs[0:10000]
songs.lyrics = songs.lyrics.str.lower()
songs.lyrics = songs.lyrics.str.replace('[^\w\s]','')
songs.lyrics = songs.lyrics.str.replace('\d+', '')
print("BEFORE: ",songs.lyrics.head())
songs.lyrics = songs.lyrics.str.replace('\n', ' ')
# print("AFTER: ",songs.lyrics.head())
songs.lyrics = songs.lyrics.map(replace_stopwords)
# print("GET....RID@@ of this. ??")
# print("Punctuation removed: ")
# print(remove_punct("GET....RID@@ of this. ??"))

# songs.lyrics = songs.lyrics.str.strip()
# songs.lyrics = songs.lyrics.dropna().apply(lambda x: [item for item in x if item not in stop_words])
# print("AFTER removing stopwords: ",songs.lyrics.head())


# In[ ]:


songs.lyrics = songs.lyrics.dropna()
songs.lyrics = songs.lyrics.str.strip()

# Filter out NANs, and all other values which are not strings: 

print("length before filtering out NaNs: ",len(songs))
songs = pd.DataFrame({'lyrics':select(songs.lyrics, str_filter)})
print("length after filtering out NaNs: ",len(songs))
songs = pd.DataFrame({'lyrics':select(songs.lyrics, not_short)})
print("length after filtering out short ones (<150 chars): ",len(songs))


# In[ ]:


# Ref: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
# nltk.download('all')
# stemmer = Porter2Stemmer()
ps = PorterStemmer()
wl=WordNetLemmatizer()
# snowball=SnowballStemmer("english")
ar1 = []
songs_stemmed = []
def process_song(text):
    stemmed_song = []
    for token in nltk.word_tokenize(str(text)):
        if len(token) > 2:
            stemmed_song.append(ps.stem(token))
    return stemmed_song

# temp_song_lyrics = ' '.join(ar1)
processed_songs = [process_song(song) for song in songs.lyrics]
processed_songs[:2]

# Bag of words: 
dictionary = gensim.corpora.Dictionary(processed_songs)
count = 0
print("Sneak Peek at the dictionary: ")
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 5:
        break
print("Size of dictionary: ")
print(len(dictionary))

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=5000)
print("Size of dictionary after filtering for most frequent words: ")
print(len(dictionary))


# In[ ]:


# Calculating how many words and their frequency of occurence: 
song_corpus = [dictionary.doc2bow(song) for song in processed_songs]
song_corpus[1]
print("Length of song corpus: ")
print(len(song_corpus))
song_xyz = song_corpus[1]

for i in range(len(song_xyz)):
    print("Word {} (\"{}\") appears {} times.".format(song_xyz[i][0],
                                                     dictionary[song_xyz[i][0]],
                                                     song_xyz[i][1]))


# In[ ]:


# ------------------------------------------------------------
# TF-IDF 

tfidf = models.TfidfModel(song_corpus)
corpus_tfidf = tfidf[song_corpus]

for song in corpus_tfidf:
    pprint(song)
    break


# In[ ]:


# Running LDA using bag of words:
lda_model = gensim.models.LdaMulticore(song_corpus, num_topics=10, id2word=dictionary, passes=2, workers=4)
# print(set(songs.genre))
# Words for each topic and occurence: 
# print(lda_model.print_topics(-1))


# In[ ]:


# Classifying sample song using our LDA model:
print("Song to be tested: ")
song_xyz=song_corpus[100]
# for i in range(len(song_xyz)):
#     print("Word {} (\"{}\") appears {} times.".format(song_xyz[i][0],
#                                                      dictionary[song_xyz[i][0]],
#                                                      song_xyz[i][1]))
# Ref: https://gist.github.com/tokestermw/3588e6fbbb2f03f89798

# multiplier=1000
def terms_to_wordcounts(terms, multiplier=2000):
    return  " ".join([" ".join(int(multiplier*song_xyz[i][1]) * [dictionary[song_xyz[i][0]]]) for i in range(len(song_xyz))])

# print([multiplier*(i[1]) for i in terms])
terms = lda_model.show_topic(4, 20)
multiplied_terms = terms_to_wordcounts(terms, 1000)
wordcloud = WordCloud(background_color="pink").generate(multiplied_terms)

# terms = lda_model.show_topic(4, 20)
# multiplied_terms = terms_to_wordcounts(terms, 1000)
# wordcloud2 = WordCloud(background_color="pink").generate(multiplied_terms)

# fig, axs = plt.subplots(1,2)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("terms1")


# In[ ]:


print("Scores for the above song for each topic (i.e. how likely is the song to belong to any of the listed topics): ")
for index, score in sorted(lda_model[song_corpus[100]], key=lambda tup: -1*tup[1]):
    print("\nThe song belongs to Topic {} with a likelihood of {}".format(index, str(round((score*100),2))),"%")
    


# In[ ]:


# vizualization  of likelihoods 

labels = ['Topic1', 'Topic2','Topic3', 'Topic4', 'Topic5', 'Topic6', 'Topic7', 'Topic8', 'Topic9', 'Topic10']
value1 = [0,0,0,14.91,68.56,0,9.06,6.98,0,0]

trace = go.Pie(labels=labels, values=value1)
py.iplot([trace], filename='basic_pie_chart')


# In[ ]:


# https://radimrehurek.com/gensim/models/ldamulticore.html#gensim.models.ldamulticore.LdaMulticore.print_topic
print("Most probable topic for the song: ")
for topic, probability in lda_model.get_document_topics(song_corpus[100], minimum_probability = 0.3):
    print("Topic",topic," with a likelihood of ",str(round((probability*100),2)),"%")


# In[ ]:


temp = lda_model.show_topic(1, 10)
terms=[]
for term in temp:
    terms.append(term[0])
curr_terms = ", ".join(terms)
print(curr_terms)


# In[ ]:


# https://gist.github.com/tokestermw/3588e6fbbb2f03f89798
for i in range(0, 10):
    temp = lda_model.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[0])
    curr_terms = ", ".join(terms)
    print("Top 10 terms for topic ",str(i), ": ", curr_terms)


# In[ ]:


# Ref: https://gist.github.com/tokestermw/3588e6fbbb2f03f89798

# multiplier=1000
def terms_to_wordcounts(terms, multiplier=2000):
    return  " ".join([" ".join(int(multiplier*i[1]) * [i[0]]) for i in terms])

# print([multiplier*(i[1]) for i in terms])
terms = lda_model.show_topic(4, 20)
multiplied_terms = terms_to_wordcounts(terms, 1000)
wordcloud = WordCloud(background_color="pink").generate(multiplied_terms)

# terms = lda_model.show_topic(4, 20)
# multiplied_terms = terms_to_wordcounts(terms, 1000)
# wordcloud2 = WordCloud(background_color="pink").generate(multiplied_terms)

# fig, axs = plt.subplots(1,2)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("terms1")

# plt.close()

