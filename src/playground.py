import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    checkpoint = "dblakely/WizardLM-13B-V1.2-fixed-tokenizer"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    print(model.embeddings)

if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()