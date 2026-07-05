# Import required modules
import re
import random
from collections import defaultdict

# Define a simple text classification model
class TextClassifier:
    def __init__(self):
        self.labels = {}
        self.vocabulary = {}

    def train(self, data):
        for text, label in data:
            if label not in self.labels:
                self.labels[label] = []
            self.labels[label].append(text)
            for word in re.findall(r'\b\w+\b', text):
                if word not in self.vocabulary:
                    self.vocabulary[word] = 0
                self.vocabulary[word] += 1

    def predict(self, text):
        max_count = 0
        predicted_label = None
        for label, texts in self.labels.items():
            count = 0
            for word in re.findall(r'\b\w+\b', text):
                if word in self.vocabulary:
                    count += 1
            if count > max_count:
                max_count = count
                predicted_label = label
        return predicted_label

# Generate synthetic data
data = []
for _ in range(1000):
    text = ' '.join(random.choice(['bug', 'feature', 'question']) for _ in range(10))
    label = random.choice(['bug', 'feature', 'question'])
    data.append((text, label))

# Train and test the model
model = TextClassifier()
model.train(data[:800])
accuracy = sum(1 for text, label in data[800:] if model.predict(text) == label) / len(data[800:])
print(f'Accuracy: {accuracy:.2f}')