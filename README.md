# Resume AI Assistant

The **Resume AI Assistant** is a tool that analyzes resumes and job descriptions to extract relevant keywords, compute a similarity score, and generate suggestions for improving the resume based on the job description. The application utilizes advanced Natural Language Processing (NLP) models to offer insights that can help individuals tailor their resumes to specific job postings.

## Features

* **PDF Resume Text Extraction**: Extract text from uploaded PDF resumes using `pdfplumber`.
* **Named Entity Recognition (NER)**: Extract key entities from the resume text such as skills, organizations, and job roles using a pre-trained BERT model (`dslim/bert-base-NER`).
* **Keyword Extraction**: Identify important keywords and entities like skills, organizations, and job roles from the resume.
* **Relevance Score**: Compute a relevance score between the resume and job description using cosine similarity based on TF-IDF vectors.
* **Suggestions Generation**: Generate actionable suggestions to improve the resume using GPT-2 based text generation, tailored to the job description.

## Technologies Used

* **Python**: Core programming language.
* **Transformers**: For Natural Language Processing, utilizing pre-trained models like BERT (NER) and GPT-2 (Text Generation).
* **pdfplumber**: For extracting text from PDF resumes.
* **sklearn**: For computing cosine similarity and generating TF-IDF vectors.
* **PyTorch**: Backend framework for transformer models.

## Installation

To run this project locally, follow the steps below:

### Prerequisites

* Python 3.8 or higher
* pip (Python package installer)

### Install Dependencies

Clone the repository:

```bash
git clone https://github.com/your-username/resume-ai-assistant.git
cd resume-ai-assistant
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following libraries:

```
transformers
torch
pdfplumber
sklearn
```

## Usage

### Running the Application

Once you have installed the dependencies, you can run the app using Streamlit to interact with the model via a web interface.

```bash
streamlit run app.py
```

### How It Works

1. **Upload Resume**: Use the interface to upload your resume in PDF format.
2. **Enter Job Description**: Input the job description of the role you're applying for.
3. **View Extracted Keywords**: The app extracts relevant keywords such as skills, organizations, and roles from both the resume and job description.
4. **View Relevance Score**: The app computes a relevance score between the resume and job description to indicate how well they match.
5. **Get Suggestions**: The app uses a GPT-2 based model to generate suggestions on how to improve your resume to match the job description better.

### Example Output

* **Extracted Keywords**: "IBM", "Machine Learning", "AI", "Data Science"
* **Relevance Score**: 44.83% (indicates the similarity between the resume and the job description).
* **Suggestions**: The model suggests improvements like, "Emphasize AI-related skills and projects more prominently in the skills section."


## Notes

* **Model Limitations**: The suggestions generation model (GPT-2) might not always provide perfectly relevant suggestions. The accuracy of suggestions can vary depending on the quality of input data.
* **Entity Extraction**: The Named Entity Recognition model may not catch all relevant entities in the resume. It's important to review the extracted entities manually.
