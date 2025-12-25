"""
Utility to load and organize code files from the workspace
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
IGNORE_DIRS = {'.venv', '__pycache__', '.git', '.pytest_cache', 'playground', '.ipynb_checkpoints'}
IGNORE_FILES = {'.DS_Store', '.gitignore'}


def get_code_structure() -> Dict:
    """
    Recursively scan workspace and organize code files into categories
    Returns a nested dict structure
    """
    structure = {
        'files': [],
        'folders': {}
    }

    for item in sorted(os.listdir(WORKSPACE_ROOT)):
        # Skip ignored items
        if item in IGNORE_DIRS or item in IGNORE_FILES:
            continue

        item_path = WORKSPACE_ROOT / item

        if item_path.is_file():
            if item.endswith(('.py', '.ipynb')):
                structure['files'].append({
                    'name': item,
                    'path': str(item_path),
                    'type': 'notebook' if item.endswith('.ipynb') else 'script'
                })
        elif item_path.is_dir():
            subfolder_structure = _scan_folder(item_path)
            if subfolder_structure['files'] or subfolder_structure['folders']:
                structure['folders'][item] = subfolder_structure

    return structure


def _scan_folder(folder_path: Path) -> Dict:
    """Recursively scan a folder"""
    structure = {
        'files': [],
        'folders': {}
    }

    try:
        for item in sorted(os.listdir(folder_path)):
            if item in IGNORE_DIRS or item in IGNORE_FILES:
                continue

            item_path = folder_path / item

            if item_path.is_file():
                if item.endswith(('.py', '.ipynb')):
                    structure['files'].append({
                        'name': item,
                        'path': str(item_path),
                        'type': 'notebook' if item.endswith('.ipynb') else 'script'
                    })
            elif item_path.is_dir():
                subfolder = _scan_folder(item_path)
                if subfolder['files'] or subfolder['folders']:
                    structure['folders'][item] = subfolder
    except PermissionError:
        pass

    return structure


def flatten_structure(structure: Dict, prefix: str = "") -> List[Tuple[str, Dict]]:
    """
    Flatten the nested structure into a list of (display_name, file_info) tuples
    Useful for dropdown/selection menus
    """
    items = []

    # Add files from current level
    for file_info in structure['files']:
        display_name = f"{prefix}{file_info['name']}"
        items.append((display_name, file_info))

    # Add files from subfolders
    for folder_name, subfolder in structure['folders'].items():
        new_prefix = f"{prefix}{folder_name}/"
        items.extend(flatten_structure(subfolder, new_prefix))

    return items


def read_file_content(file_path: str) -> str:
    """Read file content"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def get_description_path(file_name: str) -> Path:
    """Get the description markdown file path for a code file"""
    desc_name = file_name.replace('.py', '.md').replace('.ipynb', '.md')
    return WORKSPACE_ROOT / 'playground' / 'descriptions' / desc_name


def load_description(file_name: str) -> str:
    """Load description for a file if it exists"""
    desc_path = get_description_path(file_name)
    if desc_path.exists():
        try:
            with open(desc_path, 'r') as f:
                return f.read()
        except Exception:
            return None
    return None


def get_summary_from_docstring(content: str) -> str:
    """Extract docstring from Python file"""
    lines = content.split('\n')
    in_docstring = False
    docstring_lines = []

    for line in lines[:50]:  # Check first 50 lines
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                quote_type = '"""' if '"""' in line else "'''"
                docstring_lines.append(line.split(quote_type)[1] if quote_type in line else "")
            else:
                docstring_lines.append(line.split(quote_type)[0])
                break
        elif in_docstring:
            docstring_lines.append(line)

    return '\n'.join(docstring_lines).strip()
