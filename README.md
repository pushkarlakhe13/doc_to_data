# Structured Data Extractor 
Structured Data Extractor is a Streamlit-based web application designed to help users to extract structured data from unstructred documents like PDFs. It creates data extraction schemas from user inputs. The application allows users to define input fields, their descriptions, and the expected results for each input. The generated schema is then passed to GPT models to extract relevant information

# Features
- User-friendly interface for defining data extraction tasks
- Customizable input fields and descriptions
- Generates Pydantic schema classes with data validation
- Displays the generated schema with proper formatting


# Installation
To set up the locally, follow these steps:

Clone the repository:
git clone https://github.com/yourusername/schema-auto-generator.git

The application should now be running on your local machine.

# Usage
- Open your browser and go to the Streamlit app URL (usually http://localhost:8501).
- Upload a PDF file (optional) for reference.
- Enter a short description of the data extraction task.
- Define the input fields and their descriptions using the provided text inputs.
- Enter an example for the model to understand.
- Provide the expected results for each input ID.
- Click on the "See generated Schema" expander to view the generated schema.
- Copy the generated schema code and use it in your projects as needed.

