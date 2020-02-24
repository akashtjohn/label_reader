# label_reader

![alt text](https://raw.githubusercontent.com/akash1729/label_reader/master/output_images/9_merged_boxes.jpg)

you  can know more about the code from the article
part 1 : https://medium.com/ak-thomas/label-reading-from-cardboard-box-using-cv2-and-deeplearning-libraries-6673f6c04b4f

part 2 : https://medium.com/ak-thomas/label-reading-from-cardboard-box-using-cv2-and-deeplearning-libraries-db767a2a8ca3

Components:

1. Hed to detect edges of the label
2. Find the text boxes using east edge detection model
3. Merge the similar boxes 
4. Pytesseract OCR on the boxes to find the text

