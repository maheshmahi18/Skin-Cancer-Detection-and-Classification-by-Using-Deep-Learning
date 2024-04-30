import google.generativeai as genai
import os

# Explicitly set the API key here
os.environ["GEMINI_API_KEY"] = "AIzaSyBl4xqbJfKXLv1cKRI97MTHyp58eXj8ORM"

# Configure generative AI with the API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create a generative model
model = genai.GenerativeModel('gemini-pro')

# Ask for a question or input from the user
s = input("What do you want to know? ")
print("Please wait...")

# Generate content
response = model.generate_content(s)

# Print the generated content
print(response.text)
