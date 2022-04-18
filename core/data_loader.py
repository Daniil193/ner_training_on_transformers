from datasets import Dataset, DatasetDict
from typing import Generator
import hydra
import os


class DataLoader:
    """
    Load data from txt files and convert that to Dataset

    :param train_tokens_path: path to file with tokens (train part)
    :type train_tokens_path: string
    :param train_labels_path: path to file with labels (train part)
    :type train_labels_path: string

    :param valid_tokens_path: path to file with tokens (valid part)
    :type valid_tokens_path: string
    :param valid_labels_path: path to file with labels (valid part)
    :type valid_labels_path: string

    :param data_separator: separator between token or labels at files
    :type data_separator: string
    """

    def __init__(
        self,
        train_tokens_path: str,
        train_labels_path: str,
        valid_tokens_path: str,
        valid_labels_path: str,
        data_separator: str,
    ):
        self.tr_t_path = train_tokens_path
        self.vld_t_path = valid_tokens_path
        self.tr_l_path = train_labels_path
        self.vld_l_path = valid_labels_path
        self.data_separator = data_separator
        self.dataset = DatasetDict()

    @staticmethod
    def read_txt_file(filepath: str) -> Generator:
        """
        Read txt file by line and return as generator

        :param filepath: path to txt file
        :type filepath: string
        :return: generator with string lines of file
        :rtype: generator
        """
        with open(filepath, "r") as f:
            for line in f:
                yield line.strip().replace("\n", "")

    def split_text(self, text: str) -> list:
        """
        Split input text by separator

        :param text: input text
        :type text: string
        :return: list of tokens from string
        :rtype: list
        """
        assert type(text) == str, text
        text = text.split(self.data_separator)
        return text

    def init_dataset(self):
        """
        Preparing data and create dataset object
        """
        parts = {
            "train": [self.tr_t_path, self.tr_l_path],
            "valid": [self.vld_t_path, self.vld_l_path],
        }

        for part in parts:
            tokens = self.read_txt_file(parts[part][0])
            labels = self.read_txt_file(parts[part][1])

            tokens = [self.split_text(i) for i in tokens]
            labels = [self.split_text(i) for i in labels]

            assert len(tokens) == len(labels), f"{len(tokens)} & {len(labels)}"

            self.dataset[part] = Dataset.from_dict({"tokens": tokens, "labels": labels})

    def get_dataset(self):
        self.init_dataset()
        return self.dataset

    def save_dataset(self, path_to_save: str):
        if len(self.dataset) == 0:
            self.init_dataset()
        self.dataset.save_to_disk(path_to_save)


@hydra.main(config_path="../config", config_name="dataload")
def main(cfg):
    print(f"Data saved at {cfg.params.folder_name_to_save}")
    DataLoader(
        os.path.join(cfg.base_path, cfg.files.p_train_tokens),
        os.path.join(cfg.base_path, cfg.files.p_train_labels),
        os.path.join(cfg.base_path, cfg.files.p_valid_tokens),
        os.path.join(cfg.base_path, cfg.files.p_valid_labels),
        cfg.params.data_separator,
               ).save_dataset(os.path.join(cfg.base_path,
                                           cfg.params.folder_name_to_save))


if __name__ == "__main__":
    main()
