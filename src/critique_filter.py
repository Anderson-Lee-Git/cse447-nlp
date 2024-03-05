# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic")
model = AutoModelForSequenceClassification.from_pretrained("McGill-NLP/roberta-large-faithcritic")

# TODO: Sequence classification