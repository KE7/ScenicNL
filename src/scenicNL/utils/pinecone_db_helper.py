import os
from scenicNL.common import VectorDB

PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
DIMENSION = 768

def get_docs(doc_dir):
    """
    Returns an array of all the content of the scenic programs stored in doc_dir
    """

    docs = []
    for root, _, files in os.walk(doc_dir):
            for file in files:
                if file.endswith('.scenic'):
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        docs.append(f.read())
    return docs



if __name__ == '__main__':
    """
    Adds data a pinecone DB from the speicified directory
    """
    index_name = 'scenic-programs-reverseprompt'
    doc_dir = 'merged_prompt_outputs'
    docs = get_docs(doc_dir)
    db = VectorDB(index_name=index_name)
    db.upsert(docs, index=db.index)
