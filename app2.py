import streamlit as st
import pickle
import re
def load_node(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
class Node:
    def __init__(self, value=None, next_nodes=None):
        self.value = value  # Store the value as a dictionary
        self.next_nodes = next_nodes if next_nodes else []  # List of next nodes

    def sort_next_nodes_by_z_score(self):
        self.next_nodes.sort(key=lambda node: node.value['z-score'], reverse=True)

    def __repr__(self):
        return f'Node(value={self.value})'

def escape_and_highlight(text,high=False):
    """Escape special characters and highlight numbers in the text."""
    text = text.replace('\\', r'\\')  # Escape the backslash
    if high:
        text = text.replace('$', r'\$')  # Escape the $ sign
    text = text.replace('\n', '<br>')  # Replace newlines with HTML line breaks
    text = re.sub(r'(\d+)', r'<span style="color:red;">\1</span>', text)  # Highlight numbers in red
    return text

def apply_custom_font(text):
    """Wrap text in a div with custom CSS styling for consistency."""
    return f"""
    <div style="
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    ">
        {text}
    </div>
    """

def display_node(node, history):
    current_node_text = escape_and_highlight(node.value['node'],True)
    current_node_text = apply_custom_font(current_node_text)
    st.markdown(f"**Current Breakdown:** {current_node_text}", unsafe_allow_html=True)

    if node.next_nodes:
        next_node_values = [escape_and_highlight(child.value['value']) for child in node.next_nodes]
        
        # CSS styling for radio buttons
        st.markdown("""
            <style>
            .custom-radio input[type="radio"] {
                display: none;
            }
            .custom-radio input[type="radio"] + label {
                font-family: 'Arial', sans-serif;
                font-size: 16px;
                padding: 8px 12px;
                margin-bottom: 5px;
                display: inline-block;
                cursor: pointer;
                border-radius: 5px;
                border: 1px solid #ddd;
                background-color: #f9f9f9;
                transition: all 0.2s;
            }
            .custom-radio input[type="radio"]:checked + label {
                background-color: #0366d6;
                color: white;
                border-color: #0366d6;
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown("**Choose an option to further break down:**")
        
        # Custom radio button implementation
        for i, value in enumerate(next_node_values):
            st.markdown(f"""
                <div class="custom-radio">
                    <input type="radio" id="option{i}" name="radio" value="{i}">
                    <label for="option{i}">{value}</label>
                </div>
            """, unsafe_allow_html=True)

        selected_option = st.session_state.get("selected_option", None)
        if st.button("Drill down"):
            selected_option = st.session_state.get("selected_option", 0)
            selected_node = node.next_nodes[selected_option]
            history.append(selected_node)
            st.experimental_rerun()

    else:
        end_report_text = escape_and_highlight(node.value['value'],True)
        end_report_text = apply_custom_font(end_report_text)
        st.markdown(f"**End Report:** {end_report_text}", unsafe_allow_html=True)

    if history and len(history) > 1:
        if st.button("Back to parent node"):
            history.pop()
            st.experimental_rerun()

def main():
    # Load the tree structure from a pickle file
    root = load_node('tree.pkl')
    if 'history' not in st.session_state:
        st.session_state.history = [root]

    current_node = st.session_state.history[-1]
    current_node.sort_next_nodes_by_z_score()
    display_node(current_node, st.session_state.history)

if __name__ == "__main__":
    st.title("RCA UI for Variable Profit")
    main()
