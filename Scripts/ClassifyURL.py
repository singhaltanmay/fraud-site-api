# loading dependencies
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

# loading count vectorizer and model
with open("E:/ML Projects/Fraud website detection API/Models/SklearnCV.vectorizer", 'rb') as handle:
    cv = pickle.load(handle)

with open("E:/ML Projects/Fraud website detection API/Models/LogisticModelDeeper.model", 'rb') as handle:
    model = pickle.load(handle)

# Join list to str function
def join(l: list):
    f=""
    for i in l:
        f+=str(i)+' '

    return f

# predict function
def predict(query: str, cv=cv, model=model, join=join):

    # initialize tokenizer and stemmer
    tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
    stemmer = SnowballStemmer('english')

    # generate tokenized query
    tokenized_query = tokenizer.tokenize(query)

    # generate stemmed query
    stemmed_query = [stemmer.stem(word) for word in tokenized_query]

    # generate final text
    final_text = join(stemmed_query)

    # vectorize
    vectorized_query = cv.transform([final_text])

    #predict and return
    out = model.predict_proba(vectorized_query)
    return round(out[0][1]*100,2)

'''
trial = 'www.youtube.com'
print(f'{predict(trial)}% chances that {trial} is fraud')

'''
