import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd

# Specify the path to the Tesseract executable (Update with your Tesseract path)
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return threshold

def detect_areas_and_text(image, templates):
    extracted_texts = []
    
    for template_path in templates:
        # Read the template
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        # Convert the input image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use template matching
        result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)

        # Get the location of the best match
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Get the coordinates of the region to be detected
        x, y = max_loc

        # Get the width and height of the template
        height, width = template.shape

        # Draw a rectangle around the detected region
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Extract text from the detected region using Tesseract OCR after preprocessing
        roi = image[y:y + height, x:x + width]
        processed_roi = preprocess_image(roi)
        text = pytesseract.image_to_string(processed_roi, config='--psm 6')

        extracted_texts.append(text.strip())
    
    return image, extracted_texts

# Specify the path to the video file in MP4 format
video_source = 'sample2.mp4'

# Specify the paths to multiple templates
template_paths = ['detect1.png', 'detect2.png', 'detect3.png']

# Specify the path to save the Excel file
excel_file_path = 'extracted_text.xlsx'

# Open the video capture object
cap = cv2.VideoCapture(video_source)

# Create an empty DataFrame with columns based on template names
df = pd.DataFrame(columns=['Frame'] + [f'Template_{i+1}' for i in range(len(template_paths))])

# Create a Matplotlib figure for displaying the result
fig, ax = plt.subplots()

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from the video stream.")
        break

    # Detect areas and extract text
    result_frame, extracted_texts = detect_areas_and_text(frame, template_paths)

    # Add the frame number and extracted text to the DataFrame
    row_data = {'Frame': cap.get(cv2.CAP_PROP_POS_FRAMES)}
    for i, text in enumerate(extracted_texts):
        row_data[f'Template_{i+1}'] = text
    df = df.append(row_data, ignore_index=True)

    # Display the result using Matplotlib
    ax.imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Extracted Texts: {extracted_texts}', fontsize=14)
    plt.show()

    # Break the loop if 'q' is pressed
    if plt.waitforbuttonpress(timeout=0.01):
        break

# Release the video capture object
cap.release()

# Save the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False)

print(f"Extracted texts saved to: {excel_file_path}")
