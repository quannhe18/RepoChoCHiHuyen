from itertools import count
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd

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
first_document_vector = tf_idf_vector[0]


doc1 = pd.DataFrame(first_document_vector.T.todense(), index =tokens_name, columns=['tfidf'])
# print(doc1)

print(first_document_vector.T.todense())
print("---------------------------------------------------------------")
print(first_document_vector.toarray())
asda