# Python Project

## Installation

To install the project, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Create Python environment
4. Run the installation command: `pip install -r requirements.txt`.

## Build index

To build the index, follow these steps:

1. Update the `.env` file with the Azure credentials and Index name.
2. Update the `file_path` in the `buildaiindex.py` script with the input data path.
3. Run the `buildaiindex.py` script  -- python buildaiindex.py

## Streamlit UI app

To run the Streamlit UI app, follow these steps:

1. Navigate to the project directory.
2. Run the Streamlit app using the command: `streamlit run app.py`. There are 3 different pages
    1. Document Query Application - for chatbot 
    2. Topic extraction tool - for topics
    3. Document similar index page 
3. Run the IndexQuery streamlit app usingthe command : 'streamlit run indexquery.py'

