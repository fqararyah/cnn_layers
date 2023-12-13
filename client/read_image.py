import cv2
 
img_file = "./ILSVRC2012_val_00018455.JPEG"
# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread(img_file, cv2.IMREAD_COLOR)

#cv2.imshow("image", img[:,:,0])

#cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
#cv2.destroyAllWindows()

print(img[0:5,0:5,2])
print(img[0:5,0:5,1])
print(img[0:5,0:5,0])
#np.savetxt('./r.txt',img[:,:,2], fmt='%i')