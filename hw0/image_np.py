from PIL import Image
import math
import numpy as np

def loadImage(name):
	img = Image.open(name)
	pixels = img.load()

	outImg = Image.new('RGB', img.size, 'black') # create a new black image

	input_array = np.array(img)
	output_array = np.array(input_array / 2, dtype = uint8)

	outImg = Image.fromarray(output_array)

	#outImg.show()
	outImg.save('Q2.jpg')
	outPixels = outImg.load() # load to pixel array

	print('origin first pixel: ', pixels[0,0])
	print('result first pixel: ', outPixels[0,0])

if __name__ ==  '__main__':
	loadImage('westbrook.jpg')
