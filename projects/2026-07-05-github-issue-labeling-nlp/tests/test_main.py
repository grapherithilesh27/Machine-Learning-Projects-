# Import required modules
import unittest
from src.main import TextClassifier

# Define test cases
class TestTextClassifier(unittest.TestCase):
    def test_train(self):
        model = TextClassifier()
        data = [('bug bug', 'bug'), ('feature feature', 'feature')]
        model.train(data)
        self.assertIn('bug', model.labels)
        self.assertIn('feature', model.labels)

    def test_predict(self):
        model = TextClassifier()
        data = [('bug bug', 'bug'), ('feature feature', 'feature')]
        model.train(data)
        self.assertEqual(model.predict('bug bug'), 'bug')
        self.assertEqual(model.predict('feature feature'), 'feature')

if __name__ == '__main__':
    unittest.main()