import cv2
import numpy as np
import math
import pytesseract
from boundbox import BoundBox


def display(img, keep_size=False):
    if not keep_size:
        img = cv2.resize(img, (500, 500))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


hed_model_path = "models/hed_pretrained_bsds.caffemodel"
hed_prototext_path = "models/deploy.prototxt"

# set the path to frozen_east_text_detection.pb
east_detector_path = "models/frozen_east_text_detection.pb"


def reduce_noise(img):

    # we use fastNlMeansDenoisingColored to reduce the noise
    noise_reduced_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return noise_reduced_image


def find_hed(image):

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

    Height, Width = image.shape[:2]

    cv2.dnn_registerLayer('Crop', CropLayer)
    net = cv2.dnn.readNet(hed_prototext_path, hed_model_path)

    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(Width, Height),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (Width, Height))
    hed = (255 * hed).astype("uint8")

    return hed


def find_biggerst_rect_contours(image):

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

    return rect


def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


def east_text_detector(image):
    # for east text detector to work we need to resize the image to multiples of 32
    (H, W) = image.shape[:2]
    new_height = 320
    new_width = 320

    # before resizing we store the original size ratios
    width_ratio = W / float(new_height)
    height_ratio = H / float(new_width)

    resized_image = cv2.resize(image, (new_width, new_height))

    # set two output layers to the network of boxes and scores
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    net = cv2.dnn.readNet(east_detector_path)
    # construct a blob from the image and then perform a forward pass
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward(output_layers)

    scores = output[0]
    geometry = output[1]

    # set minimum confidence of each text box we find, you can fine tune it accordingly but for now we will go with 0.5
    min_confidence = 0.2
    [boxes, confidences] = decode(scores, geometry, min_confidence)

    # we need a threshold for NMS. We use 0.4 as default
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, min_confidence, nms_threshold)

    box_list = []
    for i in indices:
        # we iterate through each index for finding each valid boxes from the list of all boxes

        box = boxes[i[0]]
        # xc, yc is the center of the box
        (xc, yc) = box[0]

        # l and b is the length and breadth of the box
        l, b = box[1][0], box[1][1]

        # find angle of box in radians
        angle = math.radians(box[2])

        """
        in the test images that I used the angle was not very accurate. Since we have already made the image
        straight using hed we can go with angle 0. But if you have a better east model with correct angle then use the
        proper angle. Since i had more accuracy on zero angle i went with that
        """
        angle = 0
        # create box using the data we have
        box = BoundBox.from_center(xc, yc, l, b, angle)

        # change the ratio of the coordinates of the box to match that of the original image.
        # here we make use of the ratios we saved before resizing the image
        box.change_ratio(width_ratio, height_ratio)

        # insert the box to the list that contains all the boxes
        box_list.append(box)

    return box_list


def main():
    img = cv2.imread("/home/wasp/WorkingDirectory/label_reader/output_images/1_original.jpg")

    noise_reduced_image = reduce_noise(img)
    # now we will resize the image to a smaller size like 500, 500
    new_h, new_w = 500, 500
    noise_reduced_image_resized = cv2.resize(noise_reduced_image, (new_h, new_w))

    height, width = img.shape[:2]

    # we keep the original ratio to the image to calculate the bounding box sizes
    height_ratio = height/new_h
    width_ratio = width/new_w

    hed = find_hed(noise_reduced_image_resized)

    rect = find_biggerst_rect_contours(hed)

    reshaped_rect = np.zeros((4, 2), dtype="int32")
    reshaped_rect[:, 0] = rect[:, 0] * width_ratio
    reshaped_rect[:, 1] = rect[:, 1] * height_ratio

    box = BoundBox.box_from_array(reshaped_rect)
    image = box.perspective_wrap(noise_reduced_image)

    box_list = east_text_detector(image)

    # now we have list of all the text boxes with the text value, the next step is to join
    # the once that are similar for that we will use box compare
    merged_box = BoundBox.merge_box(box_list, dx=1.2)

    text_fields = []
    for m_box in merged_box:
        # we will scale the box slightly to make it very little bigger than the text.
        # this will be useful when we use the OCR, we increase the size by 0.008 times original box
        m_box.scale_box(1.008, 1.008)

        # we will crop out the image for the bound box. crop image function will take an image as input
        # and returns the cropped image as per the dimensions of the bounding box
        cropped = m_box.crop_image(image)

        # now we can run the image to string function of pytesseract
        text = pytesseract.image_to_string(cropped)

        # we will set the text we found inside the box
        m_box.text_value = text
        text_fields.append(text)

    # for k in merged_box:
    #     k.draw_box(image)
    #
    # display(image)


main()


