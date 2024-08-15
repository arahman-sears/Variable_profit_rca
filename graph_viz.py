import tkinter as tk
from tkinter import messagebox

class Node:
    def __init__(self, value=None, next_nodes=None):
        self.value = value  # Store the value as a dictionary
        self.next_nodes = next_nodes if next_nodes else []  # List of next nodes

    def __repr__(self):
        return f'Node(value={self.value})'


class TreeUI(tk.Tk):
    def __init__(self, root_node):
        super().__init__()
        self.title("Tree Navigator")
        self.geometry("400x300")
        
        self.current_node = root_node
        self.previous_nodes = []

        self.value_label = tk.Label(self, text="", font=("Arial", 14))
        self.value_label.pack(pady=10)

        self.node_listbox = tk.Listbox(self, selectmode=tk.SINGLE, font=("Arial", 12))
        self.node_listbox.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.node_listbox.bind('<<ListboxSelect>>', self.on_select)

        self.back_button = tk.Button(self, text="Back", command=self.go_back)
        self.back_button.pack(pady=10)

        self.update_ui()

    def update_ui(self):
        """Update the UI elements based on the current node."""
        self.value_label.config(text=f"Current Node: {self.current_node.value}")
        self.node_listbox.delete(0, tk.END)

        if self.current_node.next_nodes:
            for i, node in enumerate(self.current_node.next_nodes):
                self.node_listbox.insert(tk.END, node.value)
        else:
            self.node_listbox.insert(tk.END, "No children")

    def on_select(self, event):
        """Handle the selection of a node from the listbox."""
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            if index < len(self.current_node.next_nodes):
                self.previous_nodes.append(self.current_node)
                self.current_node = self.current_node.next_nodes[index]
                self.update_ui()

    def go_back(self):
        """Go back to the previous node."""
        if self.previous_nodes:
            self.current_node = self.previous_nodes.pop()
            self.update_ui()
        else:
            messagebox.showinfo("Info", "No previous node to go back to.")


# Example usage:
if __name__ == "__main__":
    # Example tree structure
    node3 = Node(value={"name": "Leaf Node 3"})
    node2 = Node(value={"name": "Leaf Node 2"})
    node1 = Node(value={"name": "Leaf Node 1"})
    root = Node(value={"name": "Root Node"}, next_nodes=[node1, node2, node3])

    app = TreeUI(root)
    app.mainloop()