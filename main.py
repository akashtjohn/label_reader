import cv2
import sys
import operator
import pickle
import numpy as np


from BoundBox import BoundBox


def display(img):
    img = cv2.resize(img, (500, 500))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class CropLayer():
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]

        self.ystart = (input_shape[2] - target_shape[2]) // 2
        self.xstart = (input_shape[3] - target_shape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batch_size, num_channels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]



hed_model_path = "models/hed_pretrained_bsds.caffemodel"
hed_prototext_path = "models/deploy.prototxt"


img = cv2.imread("test_images/original_new.jpg")
height, width, channel = img.shape
# we use fastNlMeansDenoisingColored to reduce the noise
noise_reduced_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# now we will resize the image to a smaller size like 500, 500
H, W = 500, 500
noise_reduced_image_resized = cv2.resize(noise_reduced_image, (H, W))

# we keep the original ratio to the image to calculate the bounding box sizes
height_ratio = height/H
width_ratio = width/W



cv2.dnn_registerLayer('Crop', CropLayer)
net = cv2.dnn.readNet(hed_prototext_path, hed_model_path)

blob = cv2.dnn.blobFromImage(noise_reduced_image_resized, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")



contours, hierarchy = cv2.findContours(hed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


biggest = None
max_area = 0
epsilon = 0.03
for i in contours:
    area = cv2.contourArea(i)
    #we check for contours with area greater than 100 because we don't need very small ones
    if area > 100:
        peri = cv2.arcLength(i, True)
        #here the value of epsilon should be fixed propely to match your enviornment
        approx = cv2.approxPolyDP(i, epsilon*peri, True)
        if area > max_area and len(approx) == 4:
            rectangle = approx
            max_area = area

#reshape the numpy array of the rectangle
rect = rectangle.reshape(4, 2)
reshaped_rect = np.zeros((4, 2), dtype="int32")
reshaped_rect[:, 0] = rect[:, 0] * width_ratio
reshaped_rect[:, 1] = rect[:, 1] * height_ratio



#new = cv2.drawContours(img, [reshaped_rect], -1, (0, 0, 255), 10)
box = BoundBox.box_from_array(reshaped_rect)
transformed_image = box.perspective_wrap(noise_reduced_image)
cropped = box.crop_image(noise_reduced_image)
display(transformed_image)
display(cropped)
print('end ohhf process')