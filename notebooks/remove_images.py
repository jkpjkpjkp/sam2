import nbformat
import re

def remove_images_from_notebook(input_path, output_path):
    # Load the notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Process each cell
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Clear all outputs (which include image outputs) and execution counts
            cell.outputs = []
            cell.execution_count = None
        elif cell.cell_type == 'markdown':
            # Remove image tags with base64 data or attachments from markdown source
            cell.source = re.sub(r'!\[.*?\]\((data:|attachment:).*?\)', '', cell.source)
        # Remove attachments (used for embedded images) from all cells
        if 'attachments' in cell:
            del cell['attachments']
    
    # Save the modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Example usage
remove_images_from_notebook('automatic_mask_generator_example.ipynb', 'output_no_images.ipynb')