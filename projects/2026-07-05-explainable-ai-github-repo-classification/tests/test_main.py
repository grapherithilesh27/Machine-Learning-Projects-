# Test the Classifier class

class TestClassifier:
    def __init__(self):
        self.classifier = Classifier()

    def test_classify(self):
        repo = Repository('A machine learning repository', ['machine learning'], ['python'], 100)
        assert self.classifier.classify(repo) == 'Machine Learning'

    def test_explain(self):
        repo = Repository('A machine learning repository', ['machine learning'], ['python'], 100)
        assert self.classifier.explain(repo) == 'This repository is classified as Machine Learning because it has machine learning in its topics.'

# Run the tests
classifier = Classifier()
test_classifier = TestClassifier()
test_classifier.test_classify()
test_classifier.test_explain()