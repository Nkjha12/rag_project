"""
Main script to demonstrate the RAG pipeline.
"""

from pipeline import RAGPipeline

if __name__ == "__main__":
    # Define your data sources here
    PDF_PATHS = ["Nitin-Jha-Resume.pdf"] # Replace with your PDF file names in the data/ folder
    WEB_URLS = ["https://study.iitm.ac.in/ds/"] # Replace with your web URLs
    YOUTUBE_URLS = ["https://www.youtube.com/watch?v=86LzMTopeGk"] # Replace with YouTube URLs

    # Create an instance of the pipeline
    rag_pipeline = RAGPipeline()

    # Run the full pipeline to ingest the data
    rag_pipeline.run_pipeline(pdf_paths=PDF_PATHS, web_urls=WEB_URLS, youtube_urls=YOUTUBE_URLS)

    # Example query
    query = "What is a large language model?"
    answer = rag_pipeline.run_query(query)
    print("\n--- Final Answer ---")
    print(answer)

    # You can run multiple queries after the pipeline is set up
    # query_2 = "What are the key benefits of RAG?"
    # answer_2 = rag_pipeline.run_query(query_2)
    # print("\n--- Second Answer ---")
    # print(answer_2)
