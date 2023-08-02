import os


def size_in_mb(file_name, human_readable=False):
    if human_readable:
        return f"{os.stat(file_name).st_size / (1024 * 1024):.2f} MB"

    return os.stat(file_name).st_size / (1024 * 1024)
