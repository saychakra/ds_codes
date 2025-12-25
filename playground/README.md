# DS Code Playground 🧪

An interactive web-based playground for exploring, understanding, and experimenting with your machine learning and data science code.

## Features

✨ **Code Navigation**
- Browse all Python scripts and Jupyter notebooks in your workspace
- Organized file explorer with folder hierarchy

📖 **Code Understanding**
- View source code with syntax highlighting
- Extract and display docstrings and summaries
- Custom markdown descriptions for each file
- Display file metadata and statistics

💻 **Code Execution**
- Interactive playground for running code snippets
- Parameter experimentation framework
- Results visualization

📊 **Learning Support**
- Understand what each code file does
- Related files and cross-references
- Learning concepts documentation
- Further reading suggestions

## Getting Started

### Installation

1. Install Streamlit in your virtual environment:
```bash
uv pip install streamlit
```

2. Run the app:
```bash
cd playground
streamlit run app.py
```

Or use the provided script:
```bash
./run.sh
```

The app will open at `http://localhost:8501`

## Project Structure

```
playground/
├── app.py                 # Main Streamlit application
├── run.sh                 # Quick start script
├── utils/
│   ├── __init__.py
│   └── code_loader.py     # File scanning and loading utilities
├── pages/                 # Future: Multi-page app
├── descriptions/          # Markdown documentation for each file
│   ├── linear_regression.md
│   ├── logistic_regression.md
│   ├── k_nearest_neighbors.md
│   └── ... (add more)
└── README.md
```

## Adding Descriptions

To add documentation for your code files:

1. Create a markdown file in `descriptions/` folder
2. Name it after your code file (e.g., `linear_regression.md` for `linear_regression.py`)
3. Use this template:

```markdown
# File Name

## What does this code do?
Brief explanation...

## Key Components
- **Component 1**: Description
- **Component 2**: Description

## Learning Concepts
- Concept 1
- Concept 2

## Algorithm Flow
Visual representation of the algorithm

## Related Files
- Link to related code

## Further Reading
- Relevant resources
```

## Features Overview

### 📂 Navigation Tab
- File selector to browse all code
- Workspace statistics
- Quick access to Python files and notebooks

### 📖 Overview Tab
- File metadata
- Extracted docstrings
- Quick summary of the code

### 💻 Code Tab
- Full source code with syntax highlighting
- Line numbers
- Download code button

### 🎮 Playground Tab
- Experiment with code
- Parameter modification
- Execution framework
- Results visualization

### 📝 Description Tab
- Custom documentation
- Learning concepts
- Usage examples
- Related resources

## Customization

### Styling
Modify the CSS in `app.py` to customize colors and layout:
```python
st.markdown("""
<style>
    /* Add custom CSS here */
</style>
""", unsafe_allow_html=True)
```

### Adding New Pages
Create files in the `pages/` directory following Streamlit's multi-page app structure.

### Extending Code Execution
Enhance the Playground tab to support:
- Parameter inputs
- Live plotting
- Data visualization
- Import specific functions from code files

## Next Steps

1. ✅ Run the app and explore your codebase
2. 📝 Add descriptions for your main files
3. 🎮 Enhance the playground with specific code execution
4. 📊 Add visualization and analysis features
5. 🎨 Customize styling and layout

## Tips for Best Results

- Keep descriptions concise but informative
- Add real usage examples in descriptions
- Include learning concepts to help revision
- Link related files for better navigation
- Update descriptions as code evolves

## Troubleshooting

**"Module not found" errors?**
- Make sure all imports in your code files are available in your environment
- Check that you're running from the correct virtual environment

**Streamlit not found?**
- Run: `uv pip install streamlit`

**Files not showing up?**
- Make sure files are in the workspace root or organized subfolders
- Check that filenames end with `.py` or `.ipynb`

## Learn More

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Data Science Stack](https://datasciencetoolbox.org/)
