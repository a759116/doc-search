#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python
import os, glob
import time
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


def load_documents():
    
    data_path = "."
    #locate data
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if (d == "data"):
                data_path = os.path.abspath(d)

    #read each text file as one document 
    files = {}
    os.chdir(data_path)
    for file in glob.glob("*.txt"):
        with open(file, 'r') as file:
            files[file.name] = file.read().replace('\n', '')
    
    dataset = pd.DataFrame.from_dict(files, orient='index', columns=['doc']).reset_index().rename(columns={"index":"file_name"})
    return dataset


# In[3]:


def simple_string_match(dataset, search_term):
    search_result = {}
    
    for index, row in dataset.iterrows():
        search_result[row['file_name']] = row['doc'].lower().count(search_term.lower())
    
    return search_result


# In[67]:


import re
def search_using_regex(dataset, search_term):
    search_result = {}
    
    for index, row in dataset.iterrows():
        search_result[row['file_name']] = len(re.findall(search_term, row['doc'].strip('\"')))
    
    return search_result


# In[25]:


def search_using_index(index, search_term):
    search_result = index.loc[search_term]

    return search_result


# In[6]:


def build_index(dataset):
    
    cv = CountVectorizer()
    
    # build a list of documents to be analyzed
    docs = []
    for index, row in dataset.iterrows():
        docs.append(row['doc'])
    
    word_count_vector = cv.fit_transform(docs)
    
    #convert word count vector to a dataframe with words as index and columns as document names
    word_count_col =  pd.DataFrame(word_count_vector.todense(), columns=cv.get_feature_names())
    word_index = pd.merge(dataset, word_count_col, left_index = True, right_index = True).drop(columns=['doc']).set_index('file_name')
    word_index.index.name = "word_index"
    word_index = word_index.T
    
    return word_index


# In[33]:


def main():
    #get user input
    search_term = input("Enter the search term: ")
    search_method = input("Search Method: 1) String Match 2) Regular Expression 3) Indexed :> ")
    s_method = int(search_method)
    
    # search_options
    search_options = {1 : simple_string_match, 2 : search_using_regex, 3 : search_using_index}
    
    dataset = load_documents()
    index = build_index(dataset)
    
    start_time = time.time()
    if (s_method == 3):
        search_result = search_using_index(index, search_term)
    else:
        search_result = search_options[s_method](dataset, search_term)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # print results in order of relevance
    print("\n\n\n", "Search Results:",  "\n\n\n")
    sorted_result = sorted(search_result.items(), key = lambda x : x[1], reverse=True)
    for file, count in sorted_result:
        print(file, " - ", count, " matches", "\n")
    print("Elasped Time: ", elapsed_time * 1000, "ms")


# In[73]:


if __name__ == "__main__":
    main()


# In[ ]:




