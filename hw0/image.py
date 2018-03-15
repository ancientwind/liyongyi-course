from PIL import Image
import math

def loadImage(name):
	img = Image.open(name)
	pixels = img.load()

	outImg = Image.new('RGB', img.size, 'black') # create a new black image
	outPixels = outImg.load() # load to pixel array

	width, height = img.size
	processPixel(pixels, outPixels, width, height)

	print('origin first pixel: ', pixels[0,0])
	print('result first pixel: ', outPixels[0,0])

	#outImg.show()
	outImg.save('Q2.jpg')

def processPixel(pixels, outPixels, width, height):
	for i in range(width):
		for j in range(height):
			outPixels[i,j] = (halfFloorRGB(pixels,i,j))

def halfFloor(num):
	return math.floor( num / 2 )

def halfFloorRGB(pixels, i, j):
	'''return half floor value of indexed [i,j] pixel'''
	r,g,b = pixels[i,j]
	return (halfFloor(r), halfFloor(g), halfFloor(b))


if __name__ ==  '__main__':
	loadImage('westbrook.jpg')
