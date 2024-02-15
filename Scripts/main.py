#from crypt import methods
from flask import *
import json, time
import ClassifyText, ClassifyURL

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    data_set = {
        'Page' : 'Home',
        'Connection' : 'Successful',
        'Time' : time.time()
    }
    json_dump = json.dumps(data_set)

    return json_dump

@app.route('/rawtext/', methods=['GET'])
def RawText_page():
    query = str(request.args.get('query')) # /rawtext/?query=raw text for nlp model
    out = ClassifyText.predict(query)
    data_set = {
        'Page' : 'Raw Text',
        'Connection' : 'Successfull',
        'Query' : query,
        'Probability' : out, 
        'Metric' : '% PROBABILITY OF BEING FRAUDLENT',
        'Time' : time.time()
    }
    json_dump = json.dumps(data_set)

    return json_dump

@app.route('/url/', methods=['GET'])
def url():
    query = str(request.args.get('query')) # /url/?query=url for nlp model
    out = ClassifyURL.predict(query)
    data_set = {
        'Page' : 'URL',
        'Connection' : 'Successfull',
        'Query' : query,
        'Probability' : out, 
        'Metric' : '% PROBABILITY OF BEING FRAUDLENT',
        'Time' : time.time()
    }
    json_dump = json.dumps(data_set)

    return json_dump

if __name__ == '__main__':
    app.run(port=7776)
