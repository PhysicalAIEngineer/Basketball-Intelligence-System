# Imports the standard Operating System
import os

# Imports the built-in pickle module for serializing (saving) and deserializing (loading)
import pickle


def read_stub(read_from_stub, stub_path):
    """
    Read cached tracking results from a stub file.
    Args:
        read_from_stub (bool): Whether to read the stub.
        stub_path (str): Path to the stub file.
    Returns:
        Cached data if available, otherwise None.
    """

    # Returns None immediately if the flag to use cached stubs is set to False.
    if not read_from_stub:
        return None

    # Returns None if no valid file path string was provided.
    if stub_path is None:
        return None

    # Returns None if the specified stub file does not exist
    if not os.path.exists(stub_path):
        return None

    # Safely opens the file in read-binary ('rb') mode
    with open(stub_path, "rb") as f:
        return pickle.load(f)


def save_stub(stub_path, stub):
    """
    Save tracking results to a stub file.
    Args:
        stub_path (str): Path to the stub file.
        stub: Data to save.
    """

    # exits early without doing anything if no valid save path is given.
    if stub_path is None:
        return

    # Extracts the target directory path from the full file path.
    directory = os.path.dirname(stub_path)

    # Creates the target directory (and any parent directories) if they do not already exist.
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Opens the target file in write-binary ('wb') mode and serializes the 'stub' object directly to disk.
    with open(stub_path, "wb") as f:
        pickle.dump(stub, f)

