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
normalizedSourcePoints = []
normalizedCurrentPoints = []

def main():
	# Now ask for input
	print('\nThis program takes an image of a room and lets you know which room you are most likely in.\n')
	print('It then computes how closely your image matches an image in the database')
	print('If it is a close enough match-- it will tell you your current camera position in relatoin to teh camera position for teh photo it matched with.\n')
	image = raw_input("Please enter the path of the image containing a room: ")

	
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
	#determine keypoints and their descriptors from teh current image
	kp1, des1 = orb.detectAndCompute(blurredOriginal,None)


	#Loop through each image in database
	for i in range(len(directories)):
		for j in range(len(images[i][0])):
			#get database image
			img2location = directories[i] + '/' + images[i][0][j]
			img2 = blurImages(img2location)
			
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

			dist = dist/10
			stdDevAng = np.std(diffAngle)/10
			stdDevSize = np.std(diffSize)/10
			#alter dist to be the dist*stdDev*stdDev
			dist = dist*stdDevSize*stdDevAng
			#update least distance if likely
			if roomMatched[i] > dist:
				roomMatched[i] = dist
				photoMatched[i] = j
	
	#get the index of the most likely room by lowest distance
	mostLikely = roomMatched.index(min(roomMatched))
	print('\n\n******OUTPUT*********')
	if (roomMatched[mostLikely]  > 30):
		print('You are most likely in room ', mostLikely + 1, ' but we cannot say this with confidence as the current photos score is: ', roomMatched[mostLikely], ', which is above our cutoff point of 30.')

	else:
		closestImageLocation = directories[mostLikely] + '/' + images[mostLikely][0][photoMatched[mostLikely]]
		
		print('You are most likely in room: ', mostLikely + 1)
		print('Closest Image: ', closestImageLocation)
		print('Distance: ', roomMatched[mostLikely], ' (the closer to zero the higher the probability you are in given room)')


		#intrinsic parameters of camera -- found using matlab
		K = np.float32([3514.89135426281, 0, 1514.08070548149, 0, 3528.06290049699, 2019.41405283900, 0, 0, 1]).reshape(3,3)
		K_inv = np.linalg.inv(K)


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
	   
	   	#Compute the fundamental matrix using ransac
	  	F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_RANSAC, 1)

 
		# Selecting only the inliers
		pts1 = src_pts[mask.ravel()==1]
		pts2 = dst_pts[mask.ravel()==1]
	
		F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, 1)

		pts1 = pts1[mask.ravel() == 1]
		pts2 = pts2[mask.ravel() == 1]
		 
		print("Fundamental matrix: \n", F)
		
		# Find Error in matrix using x'^t(F)x = 0
		pt1 = np.array([[pts1[0][0]], [pts1[0][1]], [1]])
		pt2 = np.array([[pts2[0][0], pts2[0][1], 1]])
		fundMatrixError =  (pt2.dot(F)).dot(pt1)
		#based off one point match
		print ("Error in fundamental matrix: ", fundMatrixError)
		 
		#Draw epilines and matching points on images
		lines = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
		lines = lines.reshape(-1,3)

		lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)

		BMEpilines = blurredOriginal.copy()
		CurrEpilines = img2.copy()

		BMEpilines = drawEpilines(BMEpilines,lines,pts1,pts2)
		CurrEpilines = drawEpilines(CurrEpilines,lines2,pts2,pts1)	 

		#Calculate Essential Matrix 
		E = K.transpose().dot(F).dot(K)


		#Decomposing matrices of E
		U, diag110, Vt = np.linalg.svd(E)
		if np.linalg.det(np.dot(U,Vt))<0:
			Vt = -Vt
		E = np.dot(U,np.dot(np.diag([1.0,1.0,0.0]),Vt))
		U, diag110, Vt = np.linalg.svd(E)
		print("Essential matrix: \n", E)

		#W is a rotation matrix that 
		W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
		 

		#normalize/ homogenize image coords by applying the camera instrinic params to each point 
		for i in range(len(pts1)):
		    normalizedSourcePoints.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
		    normalizedCurrentPoints.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))
		 

		#deetrmine which of the 4 possible relative camera posistions is teh correct one 

		# First possible rotational matrix value
		R = np.dot(U, W)
		R = np.dot(R, Vt)
		# Firts possible translation vector value 
		T = U[:, 2]
		if (not checkInFront(R, T)):
		    # Second choice: T = -u3
		    T = - U[:, 2]
		    if (not checkInFront(R, T)):
		        #R = U * Wt * Vt, T = u3
		        R = U.dot(W.T).dot(Vt)
		        T = U[:, 2]
		        if not checkInFront(R, T):
		        	#Only possibility
		            T = - U[:, 2]
		 

		yR1 = -math.sin([2][0])
		xR1 = math.atan2(R[2][1]/math.cos(yR1), R[2][2]/math.cos(yR1))
		zR1 = math.atan2(R[1][0]/math.cos(yR1), R[0][0]/math.cos(yR1))

		# Decompose rotational matrix
		rotationX = math.atan2(R[2][1], R[2][2])
		rotationY = -math.asin((R[2][0]))
		rotationZ = math.atan2(R[1][0], R[0][0])

		print('xrotation1: ', xR1*180/np.pi, ' yrotation1: ', yR1*180/np.pi, ' zrotation: ', zR1*180/np.pi)

		#Output
		print("Rotation matrix: ", R)	
		print("Translation vector: ", T)
		
		cv2.imshow("Current Image", BMEpilines)
		cv2.imshow("Best Matched Image", CurrEpilines)
		k = cv2.waitKey(0)



#function that takes in a rotational matrix and a translation vector
#returns true if both sets of points when translated to 3d are in front of both cameras 
# else returns false
def checkInFront(rot, trans):
    # loop through the tuples of points to get 3d points 
    for first, second in zip(normalizedSourcePoints, normalizedCurrentPoints):
    	#calculate the z value for the 3d point
        z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], first)
        #point 3d is just being checked fro its z position, but teh array is used in calculating secondPt3d
        firstPoint3d = np.array([first[0] * z, first[1] * z, z])

        z1 = np.dot(rot[:,0] - first[0]*rot[:,2], trans.transpose()) / np.dot(rot[:,0] - first[0]*rot[:,2], second.transpose())
        #second point is calculated using the Rtranspose * first point - Rtranspose * translation
        secondPoint3d = np.array([second[0] * z1, second[1] * z1, z1])

        #check if either of the z points are negative ie the point is behind the camera pose
        if firstPoint3d[2] < 0 or secondPoint3d[2] < 0:
            return False
 
    return True
 

#Blurs the image to create better found points and matched points
def blurImages(image):
	
	#convert image to grayscale
	img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

	#MEDIAN
	median = cv2.medianBlur(img,3)

	return median

#Draws teh epilines onto the srcImage as well as the matched points 
def drawEpilines(srcImg, linesToAdd,srcPts,currPts):
	srcPts = np.int32(srcPts)
	currPts = np.int32(currPts)
	n,width = srcImg.shape
	srcImg = cv2.cvtColor(srcImg, cv2.COLOR_GRAY2BGR)
	for line,srcPt,curPt in zip(linesToAdd,srcPts,currPts):
		x0,y0 = map(int, [0,-line[2]/line[1]])
		x1,y1 = map(int, [width,-(line[2]+line[0]*width)/line[1] ])
		cv2.line(srcImg, (x0,y0), (x1,y1), (0, 255, 0),1)
		cv2.circle(srcImg,tuple(srcPt), 10, (255, 0, 0), -1)
	return srcImg



main()