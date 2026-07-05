# Unit tests for the GNN implementation
import unittest
from src.main import gnn_layer

class TestGNN(unittest.TestCase):
    def test_gnn_layer(self):
        graph = {
            'A': ['B', 'C'],
            'B': ['A', 'D'],
            'C': ['A', 'D'],
            'D': ['B', 'C']
        }
        node_features = {
            'A': [10, 20, 5],
            'B': [15, 30, 10],
            'C': [8, 15, 3],
            'D': [12, 25, 8]
        }
        updated_features = gnn_layer(node_features, graph)
        self.assertEqual(len(updated_features), 4)

if __name__ == '__main__':
    unittest.main()