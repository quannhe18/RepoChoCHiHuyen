# -*- coding: utf-8 -*-
"""
Author: Kuan 


"""

from itertools import count
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd
import numpy as np


corpus = [
 'Gonna be wild to watch all the people engaging in climate in change denialism right now effortlessly switch over to bl… ',
 'RT @lamphieryeg: Indigenous rights. LGBTQ. Racism. Climate change Climate Climate. Orange Man Bad.']


# Create the Document Term Matrix and parser the corpus
count_vector = CountVectorizer()
# count matrix 
parse_matrix = count_vector.fit_transform(corpus)

# Get data of tokens in corpus

#Token data | trả về list số lần xuất hiện của từ trong các đoạn văn bản
token_value = parse_matrix.T.todense()

# Token name
tokens_name = count_vector.get_feature_names()

# Debug time
df_token = pd.DataFrame(token_value, index= tokens_name)

# Tính toán df, idf .
#TODO: Nghiên cứu đoạn này để hiểu rõ về cách lấy tfidf
tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(parse_matrix)
tf_idf_vector = tfidf_transformer.transform(parse_matrix)

#get dataframe show data tf_idf for corpus 1
document_vector = tf_idf_vector.toarray()


# doc1 = pd.DataFrame(first_document_vector.T.todense(), index =tokens_name, columns=['tfidf'])
# print(doc1)

# dece_matrix = first_document_vector.T.todense()
# print("---------------------------------------------------------------")
# to_array_matrix = first_document_vector.toarray()

# doc_tf_idf= doc1.sort_values(by=["tfidf"], ascending=False)

vector_tfidf_corpus1 = document_vector[0]
vector_tfidf_corpus2 = document_vector[1]

def caculate_cosine(A, B):
    '''
    Caculate the cosine similarity to caculate the similar of 2 document
    in corpus
    '''
    # cosin = A.B / |A|.|B|
    
    multi_A_B = np.dot(B, A)
    decomination = np.linalg.norm(A)* np.linalg.norm(B)
    return multi_A_B/decomination

similar = caculate_cosine(vector_tfidf_corpus1, vector_tfidf_corpus2)

print(similar)
    
    
    

    