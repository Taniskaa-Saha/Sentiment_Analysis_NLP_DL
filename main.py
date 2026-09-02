import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load train, test, and validation datasets
train_df = pd.read_csv(
    "emotions/train.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

test_df = pd.read_csv(
    "emotions/test.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

val_df = pd.read_csv(
    "emotions/val.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)


# Separate text and emotion labels
train_text = train_df["text"]
train_labels = train_df["emotion"]

test_text = test_df["text"]
test_labels = test_df["emotion"]

val_text = val_df["text"]
val_labels = val_df["emotion"]


# Display unique emotion labels
label_name = train_df["emotion"].unique()
print(label_name)


# Display examples from the anger class
anger = train_df[train_df["emotion"] == "anger"]

# Display the complete text without truncation
pd.set_option("display.max_colwidth", None)

print(anger.head())


# Display a particular text entry
print(train_df.loc[12, "text"])


# Check for missing values
print("Missing values in train_df:")
print(train_df.isnull().sum())
print("\nMissing values in test_df:")
print(test_df.isnull().sum())
print("\nMissing values in val_df:")
print(val_df.isnull().sum())


# Count the number of samples in each emotion class
train_df["emotion"].value_counts()


# Calculate the percentage of each emotion class
train_df["emotion"].value_counts(normalize=True) * 100


# Visualize emotion distribution
sns.countplot(
    x="emotion",
    data=train_df,
    order=train_df["emotion"].value_counts().index
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()


# Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_tokens = 10000

tokenizer = Tokenizer(
    num_words=max_tokens,
    oov_token="<unk>"
)

# Fit tokenizer only on training text
tokenizer.fit_on_texts(train_df["text"])

# Convert text into sequences
train_sequence = tokenizer.texts_to_sequences(train_df["text"])
test_sequence = tokenizer.texts_to_sequences(test_df["text"])
val_sequence = tokenizer.texts_to_sequences(val_df["text"])


# Padding
max_length = 50

train_sequence_padded = pad_sequences(
    train_sequence,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

test_sequence_padded = pad_sequences(
    test_sequence,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

val_sequence_padded = pad_sequences(
    val_sequence,
    maxlen=max_length,
    padding="post",
    truncating="post"
)


# Encode labels
label_mapping = {
    "sadness": 0,
    "anger": 1,
    "love": 2,
    "surprise": 3,
    "fear": 4,
    "joy": 5
}

train_labels = train_labels.map(label_mapping)
test_labels = test_labels.map(label_mapping)
val_labels = val_labels.map(label_mapping)


# Convert labels to NumPy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)


# Number of classes
num_classes = len(np.unique(train_labels))

print(f"Number of classes: {num_classes}")

# Vocabulary and sentence length
print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Max sentence length: {train_sequence_padded.shape[1]}")

# Check shapes
print("Train sequence shape:", train_sequence_padded.shape)
print("Test sequence shape:", test_sequence_padded.shape)
print("Validation sequence shape:", val_sequence_padded.shape)

print("Train labels shape:", train_labels.shape)
print("Test labels shape:", test_labels.shape)
print("Validation labels shape:", val_labels.shape)


#Using Class Weights fot imbalanced datasets
#class weights add weights, giving more importance to minority classes and helping the model learn better from imbalanced datasets.
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

#Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)