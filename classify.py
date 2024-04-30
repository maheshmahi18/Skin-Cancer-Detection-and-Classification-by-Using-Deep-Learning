import google.generativeai as genai

# Configure generative AI with your Gemini API key
genai.configure(api_key="AIzaSyBl4xqbJfKXLv1cKRI97MTHyp58eXj8ORM")

# Function to generate response from Gemini
def generate_response(input_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_text)
    return response.text

# Function to evaluate cancer type and provide overview and characteristics
def evaluate_cancer_type(cancer_type):
    prompt = f"Evaluate {cancer_type} cancer"
    response = generate_response(prompt)
    return response

# Specify the file path to read the predicted cancer type
output_file = r'D:/Visual Studio Code/Projects/Skin_Cancer_Detection/Files/predicted_cancer_type.txt'

# Read the predicted cancer type from the text file
with open(output_file, 'r') as f:
    predicted_cancer_type = f.readline().split(":")[1].strip()

# Evaluate predicted cancer type
evaluation = evaluate_cancer_type(predicted_cancer_type)
print("Cancer Evaluation:")
print(evaluation)