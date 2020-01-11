import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('all')

import numpy
import tflearn
import tensorflow
import random



import json

with open('intents.json') as file:
	data = json.load(file)
print(data)

words = []
labels = []
docs = []

for intent in data["intents"]: #going to loop through all the dictionaries for us
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs.append(pattern)

		if intent["tag"] not in labels:
			labels.append(intent["tag"])
		



