import os
import sys
# Directly set the project root directory
# project_root = "D:/DS Course/AI - Guided Projects/Guided Project - Session 4 - RAG in VS Code/RAG-Application"
project_root = "E:/Training/Atomcamp/DS6_Bootcamp/Sessions/Guiede_Projects/mlops"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
# sys.path.append('E:/Training/Atomcamp/DS6_Bootcamp/Sessions/Guiede_Projects/mlops')
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from custom_logging import logger
from custom_exception import CustomException

def load_documents(file_path: str):
    """
    Load documents from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: List of documents.
    """
    try:
        logger.info("Loading Documents ...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info("Documents loaded successfully from %s", file_path)
        return documents
    except Exception as e:
        raise CustomException(e, sys)

def split_documents(documents: list, chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Split documents into smaller chunks.

    Args:
        documents (list): List of documents.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        logger.info("Documents split into chunks successfully")
        return texts
    except Exception as e:
        raise CustomException(e, sys)
    
    
if __name__ == "__main__":
    # file_path = os.path.join("data", "sample.pdf")
    file_path = os.path.join(project_root, "data", "a_thousand_splendid_sun.pdf")
    documents = load_documents(file_path)
    texts = split_documents(documents)


