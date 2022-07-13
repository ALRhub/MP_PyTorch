"""
    Utilities of string operation and printing stuff
"""


def print_line(char: str = "=", length: int = 60,
               before: int = 0, after: int = 0) -> None:
    """
    Print a line with given letter in given length
    Args:
        char: char for print the line
        length: length of line
        before: number of new lines before print line
        after: number of new lines after print line

    Returns: None
    """

    print("\n" * before, end="")
    print(char * length)
    print("\n" * after, end="")
    # End of function print_line


def print_line_title(title: str = "", middle: bool = True, char: str = "=",
                     length: int = 60, before: int = 1, after: int = 1) -> None:
    """
    Print a line with title
    Args:
        title: title to print
        middle: if title should be in the middle, otherwise left
        char: char for print the line
        length: length of line
        before: number of new lines before print line
        after: number of new lines after print line

    Returns: None
    """
    assert len(title) < length, "Title is longer than line length"
    len_before_title = (length - len(title)) // 2 - 1
    len_after_title = length - len(title) - (length - len(title)) // 2 - 1
    print("\n" * before, end="")
    if middle is True:
        print(char * len_before_title, "", end="")
        print(title, end="")
        print("", char * len_after_title)
    else:
        print(title, end="")
        print(" ", char * (length - len(title) - 1))
    print("\n" * after, end="")
    # End of function print_line_title


def print_wrap_title(title: str = "", char: str = "*", length: int = 60,
                     wrap: int = 1, before: int = 1, after: int = 1) -> None:
    """
    Print title with wrapped box
    Args:
        title: title to print
        char: char for print the line
        length: length of line
        wrap: number of wrapped layers
        before: number of new lines before print line
        after: number of new lines after print line

    Returns: None
    """

    assert len(title) < length - 4, "Title is longer than line length - 4"

    len_before_title = (length - len(title)) // 2 - 1
    len_after_title = length - len(title) - (length - len(title)) // 2 - 1

    print_line(char=char, length=length, before=before)
    for _ in range(wrap - 1):
        print(char, " " * (length - 2), char, sep="")
    print(char, " " * len_before_title, title, " " * len_after_title, char,
          sep="")

    for _ in range(wrap - 1):
        print(char, " " * (length - 2), char, sep="")
    print_line(char=char, length=length, after=after)
    # End of function print_wrap_title


class BColors:
    """
    Colors
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def warn(warn_str: str):
    """
    Print a warning string in console
    Args:
        warn_str: string to be printed

    Returns:
        None
    """
    print(f"{BColors.WARNING}Warning: " + warn_str + f"{BColors.ENDC}")


def error(error_str: str):
    """
    Print an error string in console
    Args:
        error_str: string to be printed

    Returns:
        None
    """
    print(f"{BColors.FAIL}Error: " + error_str + f"{BColors.ENDC}")
