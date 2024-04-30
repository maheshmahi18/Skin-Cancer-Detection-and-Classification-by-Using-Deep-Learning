# Skin Cancer Detection and Classification

This project provides a deep learning-based approach for detecting and classifying skin cancer using the HAM10000 dataset. It includes scripts for data processing, training a convolutional neural network (CNN), and evaluating images with Google Cloud Vision. Additionally, generative AI is used to evaluate and provide insights into the predicted cancer type.

## Table of Contents
   1. Features
   2. Prerequisites
   3. Installation and Setup
   4. Usage
   5. Contributing
   6. License


## Features
- Data processing and augmentation with TensorFlow/Keras.
- Convolutional neural network (CNN) for skin cancer detection.
- Google Cloud Vision for content safety and object detection.
- Generative AI for cancer type evaluation.
- Support for saving and loading trained models.


## Prerequisites
Ensure you have the following installed before running the code:
- Python 3.7 or higher
- TensorFlow
- Pandas
- scikit-learn
- Google Cloud Vision API
- Google Generative AI API
- [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)


## Installation and Setup
1. Clone the repository:

   git clone https://github.com/maheshmahi18/Skin-Cancer-Detection-and-Classification-by-Using-Deep-Learning.git
   cd skin-cancer-detection
   
2. Install the dependencies:

   pip install -r requirements.txt

3. Google Cloud Vision Setup:

   Obtain your Google Cloud Vision API key and save it to a JSON file.

   Set the environment variable GOOGLE_APPLICATION_CREDENTIALS to point to this JSON file.

4. Google Generative AI Setup:

   Get your Gemini API key and set the GEMINI_API_KEY environment variable with the key.

5. Dataset Preparation:

   Download the HAM10000 dataset and ensure the metadata CSV and image files are placed in the appropriate directory (Files/HAM).
   Ensure the directory structure matches the paths specified in the code.


## Usage
   
   ## Training the CNN Model
   
       To train the CNN model for skin cancer classification, run the following command: python data.py

       This script preprocesses the data, creates data generators, defines a CNN model, and trains it for 10 epochs. The trained model is saved to a specified path for later use.
  
   ##  Evaluating Images for Skin Cancer

       To evaluate an image for skin cancer and predict the cancer type, run: python evaluate.py

       This script uses Google Cloud Vision to check the content safety of the image and then predicts the cancer type using the trained CNN model. The predicted result is saved to a text file.

   ## Generating Insights on Cancer Types

      To generate insights and evaluation for the predicted cancer type, run: python classify.py

      This script uses Google Generative AI to generate detailed information about the predicted cancer type and provides an overview of its characteristics.


## Contributing

Contributions are welcome! If you'd like to contribute, please submit a pull request with your changes or suggestions. Ensure that your code follows best practices and includes appropriate comments and documentation.


## License

[MIT License](LICENSE)
This project is licensed under the MIT License - see the LICENSE file for more details.
