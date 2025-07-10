"""
This module provides utility functions for file operations.

Functions:
    get_file_names: Collects and returns file names in a given directory filtered by file types.
"""

import os
from typing import List, Union, Optional

import os
from pathlib import Path
from typing import List, Union, Optional


def get_file_names(
    directory_path: Union[str, Path],
    full_path: bool = False,
    recursive: bool = False,
    file_types: Optional[Union[str, List[str]]] = None,
    file_type_inclusive: bool = True,
) -> List[str]:
    """
    Collects and returns file names in a given directory filtered by file types.

    Args:
        directory_path (str or Path): The target directory to list files from.
        full_path (bool): Whether to return the full path of the files.
        file_types (str or list): File extension or list of extensions to filter by
        (e.g., '.txt' or ['.txt', '.csv']).
        recursive (bool): Whether to search directories recursively. Default is False.
        file_type_inclusive (bool): Whether to include or exclude the specified file types.
                                    Default is True (inclusive).

    Returns:
        list: File names or full paths to the files in the directory that match the file types,
        or an empty list if none.

    Example:
        >>> get_file_names("C:/example_dir", full_path=True, file_types=[".txt", ".csv"], recursive=True)
        ['C:/example_dir/file1.txt', 'C:/example_dir/subdir/file2.csv']

        >>> get_file_names("C:/example_dir", file_types=".py", recursive=False, file_type_inclusive=False)
        ['file1.txt', 'file2.csv', 'file3.md']
    """
    if not os.path.isdir(directory_path):
        return []

    if isinstance(file_types, str):
        file_types = [file_types]

    def match_file_type(file_name: str) -> bool:
        if file_types is None:
            return True
        if file_type_inclusive:
            return any(file_name.endswith(ext) for ext in file_types)
        else:
            return not any(file_name.endswith(ext) for ext in file_types)

    matched_files = []
    for root, dirs, files in os.walk(directory_path):
        matched_files.extend(
            [
                (
                    os.path.join(root, f)
                    if full_path
                    else os.path.relpath(os.path.join(root, f), directory_path)
                )
                for f in files
                if match_file_type(f)
            ]
        )
        if not recursive:
            break

    return matched_files


def main():
    dir_path = r"C:\github\Bot0_Release1\backend"
    file_list = get_file_names(
        dir_path,
        full_path=True,
        recursive=True,
        file_types=".py",
        file_type_inclusive=True,
    )
    for file in file_list:
        print(file)


if __name__ == "__main__":
    main()
