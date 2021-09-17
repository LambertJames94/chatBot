import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle

#opens and loads the intents json file
with open("intents2.json") as file:
	data = json.load(file)	
	words = []
	dataTag = []
	words2 = []
	patTag = []
	#loops through the intents eg james, john, peter
	for intent in data["intents"]:
		#stems the word to the root in order to better train and more accurate
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			#adds root words to list
			words.extend(wrds)
			words2.append(wrds)
			#matches the pattern to the tag eg hello is in grettings or james is in james
			patTag.append(intent["tag"])
			#gets the tags of each intent
			if intent["tag"] not in dataTag:
				dataTag.append(intent["tag"])
	#removes duplicates from list to find the length of the list and also ignores ?
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	dataTag = sorted(dataTag)
################################################################
#Sends words to the bag of words and converts to 0 or 1 if present
	training = []
	output = []
	out_empty = [0 for _ in range(len(dataTag))]

	for x, doc in enumerate(words2):
		bag = []
		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row = out_empty[:]
		output_row[dataTag.index(patTag[x])] = 1
		#lists containg words 0 and 1	
		training.append(bag)
		output.append(output_row)
	#converts to numpy array
	training = numpy.array(training)
	output = numpy.array(output)
#################################################################
#Creates the neural network
#resets the graph data
tf.compat.v1.reset_default_graph()
#defines input shape and sets to how many words
net = tflearn.input_data(shape=[None, len(training[0])])
#layer
net = tflearn.fully_connected(net, 8)
#layer
net = tflearn.fully_connected(net, 8)
#output layer with probabilities
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
#Trains the model
#Sets the epoch's and batch size
model.fit(training, output, n_epoch=1, batch_size=500, show_metric=True)
#Save the model
model.save("model.tflearn")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
	return numpy.array(bag)

def chat():
	print("Start talking with the bot (type quit to stop)!")
	while True:
		inp = input("You : ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)])
		results_index = numpy.argmax(results)
		tag = dataTag[results_index]

		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = tg['responses']

		print(random.choice(responses))
chat()