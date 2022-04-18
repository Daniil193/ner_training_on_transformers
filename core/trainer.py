from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from datasets import load_metric, load_from_disk
from typing import Any
import numpy as np
import warnings
import hydra
import os

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Train model for NER task

    :param dataset: processed dataset for training
    :type dataset: Union[DatasetDict, Dataset....]
    :param model_checkpoint: name or path of model checkpoint for train
    :type model_checkpoint: string
    :param col_tokens_name: column name in dataset with tokens
    :type col_tokens_name: string
    :param col_labels_name: column name in dataset with labels
    :type col_labels_name: string
    """

    def __init__(
        self,
        dataset: Any,
        model_checkpoint: str,
        col_tokens_name: str = "tokens",
        col_labels_name: str = "labels",
    ):

        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.col_tokens_name = col_tokens_name
        self.col_labels_name = col_labels_name
        self.mapper = self.__get_mapper()
        self.metric = load_metric("seqeval")
        self.tokenizer = self.__init_tokenizer()
        self.model = self.__init_model()
        self.data_collator = self.__init_data_collator()

    def __get_mapper(self):
        """
        Create mapper for converting labels to ids from data

        get_token_counts - init mapper
        sort_mapper - sort mapper by label frequency in descending
        get_label2id - set ids for each label
        """
        mapper = dict()

        def get_token_counts(example):
            for label in example[self.col_labels_name]:
                if label not in mapper:
                    mapper[label] = 1
                else:
                    mapper[label] += 1

        def sort_mapper(mapper):
            return {
                k: v
                for k, v in sorted(
                    mapper.items(), key=lambda item: item[1], reverse=True
                )
            }

        def get_label2id(mapper):
            return {k: v for v, k in enumerate(mapper.keys())}

        self.dataset.map(get_token_counts)
        mapper = sort_mapper(mapper)
        mapper = get_label2id(mapper)
        return mapper

    def __init_model(self):
        id2label = {int(v): k for k, v in self.mapper.items()}
        return AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, id2label=id2label, label2id=self.mapper
        )

    def __init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_checkpoint)

    def __init_data_collator(self):
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def __preprocess_dataset(self):
        """
        Align labels with tokens after tokenizing

        align_labels_with_tokens - adding -100 at start and end of label sequences
        tokenize_raw - tokenize each word, depends on tokenizer
        tokenize_and_align_labels - process dataset and return a new one
        """

        def align_labels_with_tokens(labels: list, word_ids: list) -> list:
            return [-100 if i is None else labels[i] for i in word_ids]

        def tokenize_raw(examples):
            return self.tokenizer(
                examples[self.col_tokens_name],
                truncation=True,
                is_split_into_words=True,
            )

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenize_raw(examples)
            all_labels = examples[self.col_labels_name]
            new_labels = []

            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            # it's important: "labels" (column name) for result dataset
            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs

        def label_mapping(example):
            return {
                self.col_labels_name: [
                    self.mapper[i] for i in example[self.col_labels_name]
                ]
            }

        self.dataset = self.dataset.map(label_mapping)
        self.dataset = self.dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

    def train_model(self, tr_args):
        """
        Init arguments and train model

        init_args - init training arguments from config file
        compute_metrics - udf for vizualizing metrics while training
        """

        def init_args(tr_args):
            return TrainingArguments(**tr_args)

        def compute_metrics(model_output):
            logits, labels = model_output
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens) and convert to labels
            true_labels = [
                [label_names[l] for l in label if l != -100] for label in labels
            ]
            true_predictions = [
                [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = self.metric.compute(
                predictions=true_predictions, references=true_labels
            )

            return {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
                "accuracy": all_metrics["overall_accuracy"],
            }

        label_names = list(self.mapper.keys())
        self.__preprocess_dataset()
        args = init_args(tr_args)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.train()

        return trainer


@hydra.main(config_path="../config", config_name="train")
def main(cfg):

    data_set = load_from_disk(os.path.join(cfg.base_path, cfg.path_to_dataset))

    trainer = ModelTrainer(data_set, cfg.model_checkpoint, "tokens", "labels")
    model = trainer.train_model(cfg.tr_params)

    model.save_model(os.path.join(cfg.base_path, cfg.tr_params.output_dir))


if __name__ == "__main__":
    main()
