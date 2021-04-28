from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

'''
   importing NMS from utils as it is faster in processing
'''


net = cv2.dnn.readNet("frozen_east_text_detection.pb")

'''Reading the frozen graph i.e Deep Convolutional Neural Network.

   This Network contains three blocks i.e Stem,Branch and Output
   layer.Stem is the backbone where features are extracted and
   later on Merged at Branch to have features of multiscale
   which is helpful in localization of text of various sizes.
   At outputs we get scores map, RBOX geometry and QUAD geometry.
   The image of whole graph is present in this github library.
    
'''


def text_detector(image):
	''' This function returns bounding box around the 
	    the text in an image.

	'''
	orig = image
	(H, W) = image.shape[:2]
	'''
    We are resizing the input image as the effective 
    receptive field is 32. In the paper 4 levels of 
    feature map are generated at effective receptive 
    field of 32,16,8 & 4
    '''

	(newW, newH) = (640, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
    
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	'''feature_fusion/Conv_7/Sigmoid is that layer which is 
	   obtained after concatinating first layer feature map
	   and third layer of branch which is follwed by 1x1 and
	   3x3 convolutions and sigmoid is applied after convolutions.
	   This layer is used for 'score map'


	   feature_fusion/concat_3 is the obtained after concatenation
	   of two layer which were obtained from feature_fusion/conv_7/sigmoid
	   after applying 1x1 convolution which is 1 channel(rotation angle)
	   and 4 channel (text boxes) and sigmoid on top of it. Both the layers
	   are concatenated to get 5 channel output. This layer is used for
	   'RBOX'



	'''


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	'''We are subracting the mean values of RGB form image. It helps
	   CNN to not affect with illumination changes in our image.
	   Apart from this BGR is converted to RGB with swap
	'''

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	'''Extracting scores and geometry from the layers.
	   shape of scores is (1, 1, 80, 160) and shape of 
	   geometry is (1, 5, 80, 160) as these outputs are 
	   1 and 5 channel.

	   80 are the no of rows and 160 are the no of columns
	'''

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		'''Here we are iterating over the no of rows.
		   Every row shape is (160,) as every row is having
		   160 ouput containing pixels score or probability
		   of lying inside the positive are i.e bounding box
		'''
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		'''We are iterating over row to read all the channels.
		   First 4 channel contains the distance of pixel in positive
		   area to the four edges of the bounding box. Last channel
		   contains the rotation angle of the pixel for the bounding box.
		'''

		
		for x in range(0, numCols):
			'''
			For every value of row we now iterate over every column values
			which is 160 in our case.
			if our score does not have sufficient probability, ignore it
			'''
			if scoresData[x] < 0.5:
				continue

			'''compute the offset factor as our resulting feature maps will
			   be 4x smaller than the input image.
			   We started with input (320,640) and output is (80,160)
			'''
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			'''extract the rotation angle for the prediction and then
			  compute the sin and cosine
			'''
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			''' use the geometry volume to derive the width and height of
			   the bounding box, as the four channle values are the distance
			   of pixel from the four edges of the bounding box
			'''
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			''' compute both the starting and ending (x, y)-coordinates for
			  the text prediction bounding box
			'''
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			''' add the bounding box coordinates and probability score to
			   our respective lists
			'''
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	'''To eliminate all the unnecessary proposals of bounding
	           box whose IOU is less than the certain threshold
	'''

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		''' draw the bounding box on the image
		'''
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
	return orig
	

'''
   sample images for testing
'''

image2 = cv2.imread('sample.png')

image3 = cv2.imread('sample1.jpeg')
image4 = cv2.imread('Capture.png')



# for i in range(0,1):
# 	for img in array:
# 		imageO = cv2.resize(img, (640,320), interpolation = cv2.INTER_AREA)
# 		imageX = imageO
# 		orig = text_detector(imageO)
# 		cv2.imshow("Text Detection", orig)
# 		cv2.imwrite("lovetext.jpg",orig)
# 		k = cv2.waitKey(30) & 0xff
# 		if k == 27:
# 			break
# cv2.destroyAllWindows()
imageO = cv2.resize(image1, (640,320), interpolation = cv2.INTER_AREA)
imageX = imageO
orig = text_detector(imageO)
cv2.imwrite("board_text.jpg",orig)
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
