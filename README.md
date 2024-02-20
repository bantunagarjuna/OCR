# OCR
Text recognition of continuous patient Health monitoring system in Real-Time.

**Objective:** Develop software to interpret data from multipara monitors via video feed, adaptable across monitor models.

**Task 1:** Detection of Data Display Areas: Identify critical data areas(HR,ECG,oxygen levels)on varying monitor screens.

**Task 2:** Detection of Displayed Values: Use OCR to digitize the numeric values of heart rate and oxygen saturation from the identified areas.

# Methodology
Python language, OpenCV, OCR, and py-tesseract were used to build the project. 

For task 1 the interested region was detected based on a sample image of the region for which image template matching was used (cv2.matchTemplate()).

For task 2, after matching the Region of Interest(ROI) in the frame of the video input by using the Tesseract OCR was used to extract the text from the entire ROI.

# Implementation
Jupiter Notebook was used to implement the project

# Result 
The output was saved in an Excel file format.

# Conclusion 
Using this project we can collect the continuous readings of the patient from the patient monitoring screen in real time. It helps us to record the readings without any human interventions.
