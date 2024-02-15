# import dependencies
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import pickle

# model net class
class ModelNet(nn.Module):
    def __init__(self, num_words, embedding_size):
        super(ModelNet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(num_words, embedding_size, max_norm=2)
        layers = [
            nn.Flatten(),
            nn.Linear(embedding_size*14544, 1)
        ]
        self.model = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        out = self.sig(self.model(x))

        return out
    
# get tokenizer, padder and tuner
with open("E:/ML Projects/Fraud website detection API/Models/tuner.tuner", 'rb') as handle:
    tuner = pickle.load(handle)
    
with open("E:/ML Projects/Fraud website detection API/Models/kerastokenizer.tokenizer", 'rb') as handle:
    tokenizer = pickle.load(handle)
        
with open("E:/ML Projects/Fraud website detection API/Models/sequencePadder.PadSequences", 'rb') as handle:
    pad_sequences = pickle.load(handle)

# predicting function
def predict(query: str, tuner=tuner, tokenizer=tokenizer, pad_sequences=pad_sequences, modelnet=ModelNet):
    
    # Extract tuner
    maxlen = tuner['maxlen']
    num_words = tuner['num_words']
    embedding_size = tuner['embedding_size']

    # initialize model
    model = modelnet(num_words, embedding_size)

    # load trained model
    check_point = torch.load("E:/ML Projects/Fraud website detection API/Models/Bestmodel.model",
                             map_location = torch.device('cpu'))
    model.load_state_dict(check_point['model'])

    # List query
    query = [query]

    # tokenize query
    tokenized_query = tokenizer.texts_to_sequences(query)

    #pad query and convert to tensor
    padded_query = pad_sequences(tokenized_query, maxlen=maxlen, padding='post')
    padded_query = torch.tensor(padded_query).long()

    # predict and return
    out = model(padded_query)
    return round(out.item()*100, 2)
    

#trial_msg =  'raw text in any format kaisa bhi text chalega even html code agar api shi chale toh behen ki choot'
#print(f'{predict(trial_msg)}% chances that {trial_msg} is fraud')

