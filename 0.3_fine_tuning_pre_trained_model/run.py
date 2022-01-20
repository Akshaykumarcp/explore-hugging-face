
# pip install datasets
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

raw_datasets
""" 
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
}) """

# access each pair of sentences in our raw_datasets object by indexing, like with a dictionary
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]

# We can see the labels are already integers, so we won’t have to do any preprocessing there. 
# To know which integer corresponds to which label, we can inspect the features of our raw_train_dataset. 
# This will tell us the type of each column:

raw_train_dataset.features
""" 
{'sentence1': Value(dtype='string', id=None), 'sentence2': Value(dtype='string', id=None), 
    'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None), 
    'idx': Value(dtype='int32', id=None)} 
    
- label is of type ClassLabel, and the mapping of integers to label name is stored in the names folder. 
- 0 corresponds to not_equivalent, and 1 corresponds to equivalent."""


