import streamlit as st
import plotly.graph_objects as go

class Node:
    def __init__(self, value=None, next_nodes=None):
        self.value = value  # Store the value as a dictionary
        self.next_nodes = next_nodes if next_nodes else []  # List of next nodes

    def __repr__(self):
        return f'Node(value={self.value})'


def create_plot(current_node):
    fig = go.Figure()

    # Positioning the nodes
    def add_node_trace(node, x=0, y=0, level=1):
        node_id = str(id(node))
        fig.add_trace(go.Scatter(x=[x], y=[y], text=[str(node.value)], mode='markers+text', textposition='bottom center'))
        child_x_start = x - 0.5 * (len(node.next_nodes) - 1)
        for i, child in enumerate(node.next_nodes):
            child_x = child_x_start + i
            child_y = y - 1
            fig.add_trace(go.Scatter(x=[x, child_x], y=[y, child_y], mode='lines'))
            add_node_trace(child, x=child_x, y=child_y, level=level+1)

    add_node_trace(current_node)

    fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500, plot_bgcolor='#222222')

    return fig


def main():
    st.title("Tree Navigation App")

    # Example tree structure
    node3 = Node(value={"name": "Leaf Node 3"})
    node2 = Node(value={"name": "Leaf Node 2"})
    node1 = Node(value={"name": "Leaf Node 1"})
    root = Node(value={"name": "Root Node"}, next_nodes=[node1, node2, node3])

    if 'history' not in st.session_state:
        st.session_state.history = [root]

    current_node = st.session_state.history[-1]

    # Display the current node value and child options
    st.write(f"**Current Node:** {current_node.value}")

    if current_node.next_nodes:
        next_node_values = [child.value for child in current_node.next_nodes]
        choice = st.radio("Choose a child node:", next_node_values, index=0)

        if st.button("Go to selected node"):
            selected_node = current_node.next_nodes[next_node_values.index(choice)]
            st.session_state.history.append(selected_node)
            st.experimental_rerun()

    if len(st.session_state.history) > 1:
        if st.button("Back to parent node"):
            st.session_state.history.pop()
            st.experimental_rerun()

    # Display the network graph
    fig = create_plot(current_node)
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()