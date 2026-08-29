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