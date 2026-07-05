# Simple Graph Neural Network implementation
# Define a graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Define node features (e.g., number of forks, watchers, contributors)
node_features = {
    'A': [10, 20, 5],
    'B': [15, 30, 10],
    'C': [8, 15, 3],
    'D': [12, 25, 8]
}

# Define the target variable (number of stars)
target = {
    'A': 50,
    'B': 75,
    'C': 30,
    'D': 60
}

# Simple GNN layer implementation
def gnn_layer(node_features, graph):
    updated_features = {}
    for node in graph:
        neighbor_features = [node_features[neighbor] for neighbor in graph[node]]
        updated_features[node] = [sum(features) / len(features) for features in zip(*neighbor_features)]
    return updated_features

# Train the GNN model
def train_gnn(node_features, graph, target):
    for _ in range(10):  # Simple iterative training
        node_features = gnn_layer(node_features, graph)
    return node_features

# Evaluate the GNN model
def evaluate_gnn(node_features, target):
    predicted_stars = {node: sum(features) for node, features in node_features.items()}
    print("Predicted stars:", predicted_stars)
    print("Actual stars:", target)

# Run the demo
node_features = train_gnn(node_features, graph, target)
evaluate_gnn(node_features, target)