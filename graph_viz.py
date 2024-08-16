import networkx as nx
import matplotlib.pyplot as plt
import pickle

class Node:
    def __init__(self, value=None, next_nodes=None):
        self.value = value  # Store the value as a dictionary
        self.next_nodes = next_nodes if next_nodes else []  # List of next nodes

    def sort_next_nodes_by_z_score(self):
        self.next_nodes.sort(key=lambda node: node.value['z-score'], reverse=True)

    def __repr__(self):
        return f'Node(value={self.value})'

def load_node(filename):
    """Load the root node from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def add_edges(graph, node, parent=None, depth=0):
    """Recursively add nodes and edges to the graph with depth as a subset attribute."""
    node_id = id(node)
    graph.add_node(node_id, label=node.value['node'], subset=depth)

    if parent is not None:
        graph.add_edge(parent, node_id)

    for child in node.next_nodes:
        add_edges(graph, child, node_id, depth + 1)

def visualize_tree(root):
    """Visualize the tree structure using networkx and matplotlib."""
    graph = nx.DiGraph()  # Directed graph for tree structure

    # Add edges to the graph starting from the root
    add_edges(graph, root)

    # Get node labels
    labels = nx.get_node_attributes(graph, 'label')

    # Use multipartite layout for a hierarchical layout
    pos = nx.multipartite_layout(graph, subset_key="subset")

    # Modify positions to achieve a top-down layout and adjust horizontal spacing
    horizontal_scaling_factor = 9  # Adjust this factor to increase or decrease horizontal spacing
    for key, (x, y) in pos.items():
        pos[key] = (y / horizontal_scaling_factor, -x / 2)  # Adjust horizontal and vertical spacing

    # Draw the graph with adjustments for figure size and font size
    plt.figure(figsize=(14, 20))  # Increase figure size
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", 
            font_size=6,  # Smaller font size
            font_weight="bold", edge_color="gray", arrows=False, 
            verticalalignment='center', horizontalalignment='center')

    plt.title("Top-Down Hierarchical Tree Visualization")
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.show()

def main():
    # Load the tree structure from a pickle file
    root = load_node('tree.pkl')

    # Visualize the tree
    visualize_tree(root)

if __name__ == "__main__":
    main()