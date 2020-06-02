# OPENING FILE

# asking for CROPPING

# then PREPROCESSING 

# convert to gray 
# adaptive thresholding


import cv2
import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
import tkFileDialog
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import rescale
from skimage.filters import threshold_local
from skimage.morphology import square,disk
from skimage.morphology import binary_opening, binary_dilation
from skimage.morphology import thin, skeletonize



#select image function
def select_image():
	#grab a reference to image panels
	#global img
	#open file and choose image
	path = tkFileDialog.askopenfilename()
	win.destroy()
		
	#if len(path)>0 :
			

	#read image using skimage
	img_start = io.imread(path)

	# SHOW RESIZED IMAGE IN CV2, i.e, SKEW INTER_AREA
	scaled = cv2.resize(img_start,(700,600),interpolation=cv2.INTER_AREA)
	cv2.imshow("Resized image", scaled)
	cv2.waitKey(0)
	saved = cv2.imwrite('data/original.jpg', scaled)
	

	# C R O P P I N G

	print("DO YOU WANT TO CROP THE IMAGE ? Press Y or N")
	option = str(raw_input())
	
	if (option == "y") or (option == "Y") :
		
		
		#destroy previous loaded image
		cv2.destroyWindow("Resized image")
		
		# initialize the list of reference points and boolean indicating
		# whether cropping is being performed or not
		refPt = []
		
		cropping = False
		def click_and_crop(event, x, y, flags, param):
			# grab references to the global variables
			global refPt, cropping, crp
			# if the left mouse button was clicked, record the starting
			# (x, y) coordinates and indicate that cropping is being
			# performed
			if event == cv2.EVENT_LBUTTONDOWN:
				refPt = [(x, y)]
				cropping = True
				
			# check to see if the left mouse button was released
			elif event == cv2.EVENT_LBUTTONUP:
				
				# record the ending (x, y) coordinates and indicate that
				# the cropping operation is finished
				refPt.append((x, y))
				
				
				cropping = False

				# draw a rectangle around the region of interest
				cv2.rectangle(crop_img, refPt[0], refPt[1], (255, 0, 0), 3)
				#cv2.imshow("To_Be_Cropped", crop_img)
				
				#we did this because refPt was not sure of appending
				crp = refPt


		# read load image
		# load the image, clone it, and setup the mouse callback function
		crop_img = scaled 
		clone = crop_img.copy()
		cv2.namedWindow("To_Be_Cropped")
		cv2.setMouseCallback("To_Be_Cropped", click_and_crop)
		
		
	
		# keep looping until the 'q' key is pressed
		while True:
			# display the image and wait for a keypress
			cv2.imshow("To_Be_Cropped", crop_img)
			
			key = cv2.waitKey(1) & 0xFF

			# if the 'r' key is pressed, reset the cropping region
			if key == ord("r"):
				crop_img = clone.copy()
			# if the 'c' key is pressed, break from the loop
			elif key == ord("c"):
				break		
		# if there are two reference points, then crop the region of interest
		# from the image and display it
		
		
		
		if len(crp) == 2:
			#we reassign crop points to reference points(refPt) so that
			#it doesn't show empty list of refPt
			refPt = crp
			roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			
			#io.imshow(roi)
			#io.show()
			#cv2.namedWindow("ROI",cv2.WINDOW_NORMAL)
			#cv2.imshow("ROI", roi)
			#cv2.waitKey(0)
		# close all open windowsfpt
		cv2.destroyAllWindows()
		
		
	
		# showing roi in cv2
		cv2.imshow("Cropped image", roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		# assign region of interest(roi) to another variable
		forward_img = roi



	elif (option == "N") or (option == "n") :
		

		# destroy previous loaded image
		cv2.destroyWindow("Resized image")
		
		# assign scaled img to same variable
		forward_img = scaled 
		

	else :
		print("INVALID INPUT , please re-run the program")
	
	# proceed using rescaled image
	

	# G R A Y S C A L E in float64
	grayscale = rgb2gray(forward_img)

	# converting gray image from float64 to uint8 for thresholding in SKimage
	gray_1 = img_as_ubyte(grayscale)
	cv2.imshow("Grayscaled Image in uint8", gray_1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# A D A P T I V E - T H R E S H O L D I N G
	# here using adaptive, that is local thresholding in SKimage
	block_size = 3
	adaptive_t = threshold_local(gray_1, block_size, method ="gaussian", mode= "wrap",
	param = 40, offset=20)
	binary_adaptive = gray_1 < adaptive_t
	
	# here image turns into BOOLEAN (binary)
	# cannot show threshold image in CV2
	# because "TypeError: mat data type = 0 is not defined"

	thres = img_as_ubyte(binary_adaptive) #for saving it in cv2 type of file (further use)
	 

	# MORPHOLOGY ( Opening > Dilation > Thinning > Skeletonization )	

	## Opening to remove white spots
	# selem is structuring element
	sel = square(2)
	# opening
	open_bin = binary_opening(binary_adaptive, sel)
	
	fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(binary_adaptive, cmap=plt.cm.gray, interpolation='nearest')
	ax[0].set_title('Thresholded Image')
	ax[0].axis('off')

	ax[1].imshow(open_bin, cmap=plt.cm.gray, interpolation='nearest')
	ax[1].set_title('Opening morphology')
	ax[1].axis('off')
	plt.show()
	#cv2.destroyAllWindows()	

	
	## Dilation
	# selem for dilation
	sel_1 = disk(2)

	dil_bin = binary_dilation(open_bin, sel_1)
	fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(open_bin, cmap=plt.cm.gray, interpolation='nearest')
	ax[0].set_title('Opening morphology')
	ax[0].axis('off')

	ax[1].imshow(dil_bin, cmap=plt.cm.gray, interpolation='nearest')
	ax[1].set_title('Dilated Image')
	ax[1].axis('off')
	plt.show()


	## Thinning

	thinning_1 = thin(dil_bin, max_iter=2)

	fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(dil_bin, cmap=plt.cm.gray, interpolation='nearest')
	ax[0].set_title('Dilated')
	ax[0].axis('off')

	ax[1].imshow(thinning_1, cmap=plt.cm.gray, interpolation='nearest')
	ax[1].set_title('Thinning of Image')
	ax[1].axis('off')

	plt.show()

	
	## SKELETONIZE 

	skel = skeletonize(thinning_1)

	fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(thinning_1, cmap=plt.cm.gray, interpolation='nearest')
	ax[0].set_title('Thinning of Image')
	ax[0].axis('off')

	ax[1].imshow(skel, cmap=plt.cm.gray, interpolation='nearest')
	ax[1].set_title('Skeletonized image')
	ax[1].axis('off')
	plt.show()
	
	

	#saving image in threshold and skeleton for further use
	skel = img_as_ubyte(skel)
	io.imsave("data/skeleton.pgm",skel)
	io.imsave("data/thresholded.pgm",thres)
	io.imsave("data/grayscale.pgm",gray_1)



#close window
def close():
	win.destroy()


#TKinter G U I - W I N D O W 
win=Tk()
win.title("MY CIRCUIT RECOGNIzER")
#win.geometry("300x300")
win.configure(background="black",height=500,width=500)
win.resizable(0,0)

#white frAME
#frame = Frame(win, bg="white")
#frame.place(relwidth=0.8, relheight=0.8, relx=0.1,rely=0.1)

#whole canvas
canvas=Canvas(win,bg="black",height=500,width=500)
canvas.pack()
desk = PhotoImage(file="cr.png")

canvas.create_image(250,250,anchor=CENTER, image=desk)



#create a button, then when pressed will trigger a file chooser
#dialog and allow the user to select an input, then 'add' the button GUI
button = Button(win, text="Select the Circuit", command=select_image)
button.config(height =1,width=1)
button.pack(side="top", fill="both", expand="no", padx="20", pady="10")


cls=Button(win,text="Exit", command=close)
cls.config(height=1, width=1)
cls.pack(side="bottom", fill="both", expand="yes", padx= "20", pady="10")

#kickoff the GUI
win.mainloop()


