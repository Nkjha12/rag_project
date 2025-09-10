## RAG Pipeline with Open-Source Tools
This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built entirely with open-source Python libraries. It can ingest data from PDF files, web pages, and YouTube videos to answer questions based on the provided content.

The pipeline uses `uv` for high-performance dependency management and is structured with Object-Oriented Programming (OOP) principles for conceptual clarity and maintainability.


## Prerequisites:

Before you begin, ensure you have the following installed on your system:
- Python 3.8+uv: 
- A modern, fast Python package installer and dependency management tool.

If you don't have uv installed, you can get it with: `pip install uv` 


## Getting Started

Follow these steps to set up the project and run the pipeline

##### 1. Clone the RepositoryIf you haven't already, clone this repository to your local machine
##### 2. Set up the Virtual EnvironmentCreate a new virtual environment using uv to isolate the project's dependencies

##### 3. Install DependenciesInstall all the required packages listed in `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

##### 4. Add Your DataPlace any PDF files you want to use inside the data/ directory.

##### 5. Run the PipelineModify the `main.py` file to include your desired PDF, web page, or YouTube video URLs. Then, execute the script from the root directory of the project:

`python main.py` 
The script will download the necessary models, process the data, and run the example query. The first run may take a few minutes as the models are being downloaded and cached

## Project Structure

```text
rag_project/
├── data/
├── .venv/
├── main.py
├── pipeline.py
├── requirements.txt
└── README.md
```


