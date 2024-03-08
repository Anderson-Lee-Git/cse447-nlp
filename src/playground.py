import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.retrievers import WikipediaRetriever


def main():
    path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/raw_gen_text.json"
    data = json.load(open(path, "r"))
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(query=data["0"])
    print(len(docs))
    print(docs[0].metadata)
    

if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()