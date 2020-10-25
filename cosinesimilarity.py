from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London","Paris Paris London"]
cv= CountVectorizer()

count_matrix = cv.fit_transform(text)

print(count_matrix.toarray()) #converts the count matrix which is in sparse form into a numpy array 

cosine_sim = cosine_similarity(count_matrix)  
print(cosine_sim)

