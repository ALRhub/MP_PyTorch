import os

from natsort import os_sorted


def get_file_names_in_directory(directory: str) -> [str]:
    """
    Get file names in given directory
    Args:
        directory: directory where you want to explore

    Returns:
        file names in a list

    """
    file_names = None
    try:
        (_, _, file_names) = next(os.walk(directory))
        if len(file_names) == 0:
            file_names = None
    except StopIteration as e:
        print("Cannot read files from directory: ", directory)
        raise StopIteration("Cannot read files from directory")
    return os_sorted(file_names)
