import pickle


def save_data(data, filename):
    """
    Saves data to a file using pickle.

    Args:
    data (any): The data to be saved.
    filename (str): The name of the file where the data will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_data(filename):
    """
    Loads data from a file using pickle.

    Args:
    filename (str): The name of the file to load data from.

    Returns:
    any: The data loaded from the file.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)


def read_pickle_data(filename):
    data = load_data(filename)

    print("\nGenerated Summary:")
    print(data['generated_summary'])

    print("\nReference Summary:")
    print(data['reference_summary'])
