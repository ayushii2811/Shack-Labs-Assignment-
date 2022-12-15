import numpy as np 
import pandas as pd
import tensorflow_hub as hub
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
#import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from operator import add

#Load the respective files of Amazon and Flipkart data
am = pd.read_csv('/content/drive/MyDrive/shack_labs/amz.csv', encoding='latin1')
fk = pd.read_csv('/content/drive/MyDrive/shack_labs/flipkart.csv', encoding='latin1')

#Create and load the model of the universal sentence encoder 
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)

#Create embeddings of the product names in Amazon
amazon_embeddings_prod_name = model(list(am.product_name))

#Extract alphanumeric art from the product specifications of Amazon
am['prod_specs'] = am.product_specifications.str.findall('([a-zA-Z0-9]+)')
am['prod_specs'] = am.prod_specs.map(lambda x: ' '.join(x) if type(x)==list else '')
#Create embeddings of the product specifications of Amazon
amazon_embeddings_prod_specs = model(list(am.prod_specs))

##Create embeddings of the product names in Flipkart
flipkart_embeddings_prod_name = model(list(fk.product_name))

#Extract alphanumeric art from the product specifications of Flipkart
fk['prod_specs'] = fk.product_specifications.str.findall('([a-zA-Z0-9]+)')
fk['prod_specs'] = fk.prod_specs.map(lambda x: ' '.join(x) if type(x)==list else '')
#Create embeddings of the product specifications of
flipkart_embeddings_prod_specs = model(list(fk.prod_specs))

score = []
indices = []

#For every product in Amazon data, calculate the similarity 
#scores of product name and product specifications and select the 
#max score and the index of that product from the Flipkart data 

for i in range(len(am)):

  similarity_prod_name = cosine_similarity(
    [amazon_embeddings_prod_name[i]],
    flipkart_embeddings_prod_name
  )

  similarity_prod_specs = cosine_similarity(
    [amazon_embeddings_prod_specs[i]],
    flipkart_embeddings_prod_specs
  )

  total_similarity = [similarity_prod_name[i]+similarity_prod_specs[i] for i in range(len(similarity_prod_name))]

  idx = np.argmax(total_similarity)
  score.append(np.max(total_similarity))
  indices.append(idx)

am['indices'] = pd.Series(indices)
am['score'] = pd.Series(score)

#Create the output file
#Since each row in the Amazon data has the index of the row which
#which has the maximum similarity based on product name and product
#product similarity, we can use it to create the output file

result_1 = pd.DataFrame(columns = ['Product name in Amazon', 'Retail Price in Amazon',
                                   'Discounted Price in Amazon',
                                   'Product name in Flipkart', 'Retail Price in Flipkart',
                                   'Discounted Price in Flipkart'])

for i in range(len(am)):
  j = am.indices.iloc[i]
  data = [[am.product_name.iloc[i], am.retail_price.iloc[i], 
           am.discounted_price.iloc[i], fk.product_name.iloc[j],
           fk.retail_price.iloc[j], fk.discounted_price[j]]]
  df = pd.DataFrame(data, columns = ['Product name in Amazon', 'Retail Price in Amazon',
                                   'Discounted Price in Amazon',
                                   'Product name in Flipkart', 'Retail Price in Flipkart',
                                   'Discounted Price in Flipkart'])
  result_1 = pd.concat([result_1, df], axis = 0, copy = False)


result_1.reset_index(inplace = True)
result_1.drop('index', axis = 1, inplace = True)

result_1.to_csv('output.csv')
!cp output.csv "/content/drive/MyDrive/shack_labs"
