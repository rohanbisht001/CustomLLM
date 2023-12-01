import json
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Function to save dataset to a JSON file
def save_to_file(dataset, filename):
    with open(filename, 'w') as file:
        json.dump(dataset, file, indent=2)

# Function to load dataset from a JSON Lines file
def load_from_file(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # Skip empty lines
            if line.strip():
                try:
                    example = json.loads(line)
                    dataset.append(example)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
    return dataset



# Function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Lowercasing
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Removing punctuation
        tokens = [char for char in tokens if char not in string.punctuation]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens
    else:
        # Handle the case where 'A' is not a string (e.g., it's a list)
        return []


# # Function to balance dataset by oversampling the minority class
# def balance_dataset(dataset):
#     # Separate examples by class
#     class_0 = [example for example in dataset if example['label'] == 0]
#     class_1 = [example for example in dataset if example['label'] == 1]

#     # Determine the class with fewer examples
#     min_class_len = min(len(class_0), len(class_1))
    
#     # Oversample the minority class
#     balanced_dataset = class_0[:min_class_len] + class_1[:min_class_len]

#     return balanced_dataset

# Load dataset from JSON Lines file
input_file = 'seed_data.jsonl'
your_dataset = load_from_file(input_file)

# Shuffle the dataset
your_dataset = shuffle(your_dataset, random_state=42)

# Apply preprocessing to text data
for example in your_dataset:
    example['prompt'] = preprocess_text(example['prompt'])
    example['completion'] = preprocess_text(example['completion'])


# # Balance the dataset
# your_dataset = balance_dataset(your_dataset)

# Split the dataset
train_set, test_set = train_test_split(your_dataset, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42)

# Save the splits to files
save_to_file(train_set, 'train_dataset.jsonl')
save_to_file(val_set, 'val_dataset.jsonl')
save_to_file(test_set, 'test_dataset.jsonl')

# Load the splits from files (just for demonstration)
loaded_train_set = load_from_file('train_dataset.jsonl')
loaded_val_set = load_from_file('val_dataset.jsonl')
loaded_test_set = load_from_file('test_dataset.jsonl')

# Check the sizes of loaded sets
print(f"Loaded Training set size: {len(loaded_train_set)}")
print(f"Loaded Validation set size: {len(loaded_val_set)}")
print(f"Loaded Test set size: {len(loaded_test_set)}")
