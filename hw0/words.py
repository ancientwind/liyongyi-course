import re
import os.path

class App():
	'''Read in file name'''
	def __init__(self, name):
		self.name = name
		print('Begin to analyse txt: {}. '.format(self.name))
	
	def process(self):
		with open(self.name, 'rt') as f:
			data = f.read()
			words = re.split(r'\s', data)
			self.countWords(words)
	
	def countWords(self,words):
		index = 0
		f = open('Q1.txt', 'at')
		wordSet = set()
		wordCount = {}
		for word in words:
			if word:
				if word not in wordSet:
					wordSet.add(word)
					wordCount[word] = 1
				else:
					wordCount[word] += 1

		for name, count in wordCount.items():
			f.write(name + ' ' + str(index) + ' ' + str(count) + '\n')			
			index += 1
		f.close()

if __name__ ==  '__main__':
	app = App('words.txt')
	app.process()
