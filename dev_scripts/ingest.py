import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
import glob


# Import config vars
with open('dev_scripts/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def run_ingest():

    # Creating an array of input files
    file_array = glob.glob("data/*.pdf")

    loader = PyPDFLoader(file_array[0])

    #if ext == "jpg" or ext == "png" or ext == "":
    #    loader = DirectoryLoader(file_path,
    #                    glob='*.jpg',
    #                    loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

#if __name__ == "__main__":
#    run_ingest(r"D:\LLM\Project_2_invoice_Extraction\Approach_3\data\invoice_1.pdf")