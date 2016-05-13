from __future__ import print_function
import cv2
import numpy as np
import pylab
import os
from os import walk
from os import path
from matplotlib import pyplot as plt
import math


directories = []
images = []
roomMatched = []
photoMatched = []

def main():
	# Now ask for input
	print('\nThis program takes an image of a room and lets you know which room you are most likely in.\n')
	image = raw_input("Enter the path of the image containing a room: ")

	
	#get current working directory
	mypath = os.getcwd()
	#append /rooms
	mypath = mypath + '/Rooms'

	#get array of directory paths aka rooms
	directories = [x[0] for x in os.walk(mypath)]
	#remove rooms directory
	del directories[0]
	rooms = len(directories)

	#set up array for greatest match 
	for i in range(rooms):
		roomMatched.insert(i, 1000000)
		photoMatched.insert(i, 100)

	#get images in each room and put into images array 
	for directory in range(len(directories)):
		files = [x[2] for x in os.walk(directories[directory])]
		images.insert(directory, files)

	
	#blur original image
	blurredOriginal = blurImages(image)

	# Initiate ORB detector
	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(blurredOriginal,None)

	for i in range(len(directories)):
		for j in range(len(images[i][0])):

			img2location = directories[i] + '/' + images[i][0][j]

			img2 = cv2.imread(img2location, cv2.IMREAD_GRAYSCALE)

			
			# find the keypoints and descriptors with ORB
			kp2, des2 = orb.detectAndCompute(img2, None)
			#kp2, des2 = orb.detectAndCompute(img2,None)
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			# Match descriptors.
			matches = bf.match(des1,des2)
			# Sort them in the order of their distance.
			matches = sorted(matches, key = lambda x:x.distance)
			#get average distance away from 10 best matching points
			dist = 0
			diffSize = []
			diffAngle = []

			for x in range(10):
				dist += matches[x].distance
				#put ten angle differences into diffAngle[]
				ang = kp1[matches[x].queryIdx].angle
				ang2 = kp2[matches[x].trainIdx].angle
				angDif = ang - ang2
				diffAngle.insert(x, angDif)
				#put ten size differences into diffSize[]
				size1 = kp1[matches[x].queryIdx].size
				size2 = kp2[matches[x].trainIdx].size
				sizeDiff = size1 - size2
				diffSize.insert(x, sizeDiff)

				###NEED TO CHANGE THIS TO RATIO LOCATION
				x1, y1 = kp1[matches[x].queryIdx].pt
				x2, y2 = kp2[matches[x].trainIdx].pt

			dist = dist/10
			stdDevAng = np.std(diffAngle)/10
			stdDevSize = np.std(diffSize)/10

			dist = dist*stdDevSize*stdDevAng
			#update least distance if likely
			if roomMatched[i] > dist:
				roomMatched[i] = dist
				photoMatched[i] = j
	
	#get the index of the most likely room by lowest distance
	mostLikely = roomMatched.index(min(roomMatched))
	print('\n\n******OUTPUT*********')
	if (roomMatched[mostLikely]  > 30):
		print('You are most likely in room ', mostLikely + 1, ' but we cannot say this with confidence.')

	else:
		closestImageLocation = directories[mostLikely] + '/' + images[mostLikely][0][photoMatched[mostLikely]]
		
		print('You are most likely in room: ', mostLikely + 1)
		print('Closest Image: ', closestImageLocation)
		print('Distance: ', roomMatched[mostLikely], ' (the closer to zero the higher the probability you are in given room)')

		
		img2 = cv2.imread(closestImageLocation, cv2.IMREAD_GRAYSCALE)

		###Plot best image with test image
		# find the keypoints and descriptors with ORB
		kp2, des2 = orb.detectAndCompute(img2,None) 
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		# Match descriptors.
		matches = bf.match(des1,des2)
		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)

		srcPts = []
		dstPts = []
		# get best 40 points from matched image and source image
		for f in range(40):
					srcPt = kp1[matches[f].queryIdx].pt
					dstPt = kp2[matches[f].trainIdx].pt
					srcPts.insert(f, srcPt)
					dstPts.insert(f, dstPt)
		
		#Fill arrays with 40 matching points 
		src_pts = np.float32([srcPts[m] for m in range(len(srcPts))])
	   	dst_pts = np.float32([dstPts[m] for m in range(len(dstPts))])
	   	#Reshape matrices to conform to findHomography method
	   	src_pts.reshape(-1,1,2)
	   	dst_pts.reshape(-1,1,2)

	   	#Find homograpy --> returns which points are good and 
		retVal, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
		print(retVal)

		#find translation in source photo  
		translationX = retVal[0][2]
		translationY = retVal[1][2]
		print('Translation x: ', translationX, ' translation y: ', translationY)

		#calculate differences in scale from matched image
		scaleX = math.sqrt((retVal[0][0] ** 2) + (retVal[1][0] ** 2))
		scaleY = math.sqrt((retVal[0][1] ** 2) + (retVal[1][1] ** 2)) 
		scaleZ = math.sqrt((retVal[0][2] ** 2) + (retVal[1][2] ** 2) + (retVal[2][2] ** 2))  
		print('ScalesX:', scaleX, ' scaleY: ', scaleY, ' scaleZ: ', scaleZ)

		rotationM = retVal.copy()
		for length in range(3):
			rotationM[length][0] = rotationM[length][0]/scaleX
			rotationM[length][1] = rotationM[length][1]/scaleY
			rotationM[length][2] = rotationM[length][2]/scaleZ

		print('Rotational Matrix: ', rotationM)

		#determine rotation changes from matched image
		rotationX = math.atan2(rotationM[2][1], rotationM[2][2])
		rotationY = math.atan2(-(rotationM[2][0]), math.sqrt((rotationM[2][1] ** 2) +  (rotationM[2][2] ** 2)))
		rotationZ = math.atan2(rotationM[1][0], rotationM[0][0])
		print('rotationx: ', rotationX, ' rotationY: ', rotationY, ' rotationZ: ', rotationZ)

		#Show image moved to position of saved image
		im_out = cv2.warpPerspective(blurredOriginal, retVal, (blurredOriginal.shape[1],blurredOriginal.shape[0]))
		img3 = cv2.drawMatches(blurredOriginal,kp1,img2,kp2,matches[:5],None, flags=2)



		#Show four images 
		cv2.imshow("Search image", blurredOriginal)
		cv2.imshow("Matched image", img2)
		cv2.imshow("Warped Source Image", im_out)
		cv2.imshow("Matches", img3)

	 	
	 	k = cv2.waitKey(0)





def blurImages(image):
	
	#convert image to grayscale
	img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)


	### USE DIFFERENT BLURING TECHNIQUES
	#MEDIAN
	median = cv2.medianBlur(img,3)
	#findEdges(median, 'median', image)

	#BILATERAL
	bilateral = cv2.bilateralFilter(img, 5, 150, 150)
	#findEdges(bilateral, 'bilateral', image)

	#GAUSSIAN
	gaussian = cv2.GaussianBlur(img,(5,5),0)
	#findEdges(gaussian, 'gaussian', image)

	return median







main()