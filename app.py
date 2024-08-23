import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import os
def load_node():
    
    # Find the first file that starts with "tree" and ends with ".pkl"
    for file in os.listdir():
        if file.startswith("tree") and file.endswith(".pkl"):
            with open(file, 'rb') as f:
                #print(f"Loading node from: {file}")
                return pickle.load(f)
    raise FileNotFoundError("No file starting with 'tree' found.")
def format_value_with_dollar_sign(key, value):
    """Format value with a dollar sign for cost, discount, and revenue-related keys."""
    if key in ['Variable_Cost', 'Variable_Revenue', 'Variable_Discount']:
        return f"${value:,.2f}"
    if isinstance(value, dict):
        formatted_dict = '<br>'.join([f"  - **{k}:** {format_value_with_dollar_sign(k, v)}" for k, v in value.items()])
        return f"{{<br>{formatted_dict}<br>}}"
    return value
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

def apply_custom_font(text, width="700px"):
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
        width: {width};
        display: inline-block;
    ">
        {text}
    </div>
    """
#@st.cache(suppress_st_warning=True)
def generate_waterfall_chart(water_fall):
    
    labels = []
    deltas = []
    colors = []
    mean_values = []
    std_values = []

    # Process data
    for category, types in water_fall.items():
        for type_name, items in types.items():
            for item_name, metrics in items.items():
                labels.append(f"{item_name}")
                delta = metrics['delta']
                mean = metrics.get('mean', 0)
                std = metrics.get('std', 0)

                mean_values.append(mean)
                std_values.append(std)

                # Determine color and delta adjustment
                if (type_name == 'Revenue' and delta >= 0):
                    colors.append('green')
                    deltas.append(delta)
                elif (type_name == 'Revenue' and delta < 0):
                    colors.append('red')
                    deltas.append(delta)
                elif (type_name in ['Expense', 'Discount'] and delta <= 0):
                    colors.append('green')
                    deltas.append(-delta)
                elif (type_name in ['Expense', 'Discount'] and delta > 0):
                    colors.append('red')
                    deltas.append(-delta)

    # Calculate cumulative profit
    cumulative = np.cumsum(deltas)
    cumulative_with_base = np.hstack(([0], cumulative))

    # Dynamically determine figure size
    # fig_width = max(15, len(labels) * 0.7)  # Adjust to control spacing between bars
    # fig_height = max(10, max(cumulative_with_base) / 10 + len(labels) * 0.5)
    fig_width=34
    fig_height=24

    # Create the waterfall plot with dynamic figure size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.bar(range(len(deltas)), deltas, color=colors)
    ax.plot(range(len(deltas) + 1), cumulative_with_base, color='black', linestyle='-', marker='o',label='Cumulative Impact on Profit')
    ax.legend(fontsize=18)
    # Label cumulative line with some spacing
    for i, (x, y) in enumerate(zip(range(len(deltas) + 1), cumulative_with_base)):
        ax.text(x, y + (1 if y >= 0 else -1), f'{y:.2f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=16, color='black')

    # Add mean and standard deviation labels
    for i, (bar, mean, std) in enumerate(zip(bars, mean_values, std_values)):
        height = bar.get_height()
        # Positioning labels above or below the bars depending on their height
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'Mean: {mean:.2f}\nStd: {std:.2f} \ndelta: {height:.2f}', ha='center', va='bottom', fontsize=16, color='darkblue')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height - 1, f'Mean: {mean:.2f}\nStd: {std:.2f} \ndelta: {height:.2f}', ha='center', va='top', fontsize=16, color='darkblue')

    # Set labels and title
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90,fontsize=18)
    ax.set_ylabel('Delta values ( w.r.t to the mean of each factor)',fontsize=18)
    ax.set_title('Waterfall Chart of Profit Impact by Category',fontsize=20)

    plt.tight_layout()
    st.pyplot(fig)
def display_node(node, history):
    # Display the current node's information
    current_node_text = escape_and_highlight(node.value['node'],True)
    current_node_text = apply_custom_font(current_node_text)
    if not node.value['value']:
        st.markdown(f"**Overall report  on  the  variable profit:** {current_node_text}", unsafe_allow_html=True)
    else:
        st.markdown(f"**Current Breakdown:** {current_node_text}", unsafe_allow_html=True)

    if node.next_nodes:
        # Display each next node's value with proper formatting
        for i, child in enumerate(node.next_nodes):
            child_text = escape_and_highlight(child.value['value'])
            child_text = apply_custom_font(child_text)
            st.markdown(f"{i + 1}. {child_text}", unsafe_allow_html=True)
            st.markdown("#### Additional Details")
            for key, value in child.value.items():
                if key not in ['node', 'value', 'water_fall','tup']:  # Skip the main and already displayed keys
                    formatted_value = format_value_with_dollar_sign(key, value)
                    st.markdown(f"- **{key}:** {formatted_value}", unsafe_allow_html=True)
                    
            with st.expander(f"Expand to view chart {i + 1} and  see how different factors are impacting difference in the variable profit ", expanded=False):
                st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
                generate_waterfall_chart(child.value['water_fall'])
                st.markdown('</div>', unsafe_allow_html=True)
        
        
        # Use st.radio to select the index of the next node
        selected_option = st.radio(
            #st.markdown("Listed are breakdown within the current state, choose an option to further break down:", unsafe_allow_html=True)
            "Listed above are breakdown within the current state, choose an option to  delve deeper to see more filtered outliers :",
            options=range(len(node.next_nodes)),
            format_func=lambda i: f"Option {i + 1}"
        )

        if st.button("Drill down (It will take time to load !)"):
            selected_node = node.next_nodes[selected_option]
            history.append(selected_node)
            st.experimental_rerun()
    else:
        end_report_text = escape_and_highlight(node.value['value'],True)
        end_report_text = apply_custom_font(end_report_text)
        st.markdown(f"**End Report:** {end_report_text}", unsafe_allow_html=True)
        for key, value in node.value.items():
        
            if key not in ['node', 'value', 'water_fall','tup']:  # Skip the main and already displayed keys
                formatted_value = format_value_with_dollar_sign(key, value)
                st.markdown(f"- **{key}:** {formatted_value}", unsafe_allow_html=True)
        with st.expander(f"Expand to view chart and see how different factors are impacting difference in the variable profit", expanded=False):
            st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
            generate_waterfall_chart(node.value['water_fall'])
            st.markdown('</div>', unsafe_allow_html=True)

    if history and len(history) > 1:
        if st.button("Go Back"):
            history.pop()
            st.experimental_rerun()

def main():
    # Load the tree structure from a pickle file
    root = load_node()
    if 'history' not in st.session_state:
        st.session_state.history = [root]

    current_node = st.session_state.history[-1]
    current_node.sort_next_nodes_by_z_score()
    display_node(current_node, st.session_state.history)

if __name__ == "__main__":
    st.title("Outlier report for Variable Profit")
    main()
