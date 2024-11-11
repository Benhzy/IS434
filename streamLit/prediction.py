import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

label_columns = [
    "ableism", "anti_religion", "harm", "homophobia", "islamophobia", "lookism",
    "political_polarisation", "racism", "religious_intolerance", "sexism", "vulgarity", "xenophobia"
]

