import pandas
import random


def random_binary():
    return random.choice([0, 1])


def make_fake_edcr_dataset(output_path: str, size: int, num_columns: int):
    """
    Creates a fake dataset of a particular size and number of columns.

    Args:
    output_path: str: The path to save the dataset to.
    size: int: The number of rows in the dataset.
    num_columns: int: The number of columns in the dataset.
    """
    data = {}

    data["pred"] = [random_binary() for _ in range(size)]
    data["target"] = [random_binary() for _ in range(size)]
    for index in range(num_columns):
        data[f"feature_{index}"] = [random_binary() for _ in range(size)]

    output = pandas.DataFrame(data)
    output.to_csv(output_path, index=False)


make_fake_edcr_dataset("data/fake_1.csv", 1000, 30)
make_fake_edcr_dataset("data/fake_2.csv", 500, 20)
make_fake_edcr_dataset("data/fake_3.csv", 10, 1)
make_fake_edcr_dataset("data/fake_4.csv", 1000, 1)
make_fake_edcr_dataset("data/fake_5.csv", 200, 1000)
