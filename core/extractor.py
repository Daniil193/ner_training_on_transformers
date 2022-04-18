from data_loader import DataLoader as dl
from transformers import pipeline
from termcolor import colored
from typing import Generator
import hydra
import os


class NerExtractor:
    """
    Labeling each token in sentence as named entity

    :param model_checkpoint: name or path to model
    :type model_checkpoint: string
    """

    def __init__(self, model_checkpoint: str):
        self.token_pred_pipeline = pipeline(
            "token-classification",
            model=model_checkpoint,
            aggregation_strategy="average",
        )

    @staticmethod
    def text_color(txt: str, txt_c: str = "blue", txt_hglt: str = "on_yellow") -> str:
        """
        Coloring part of text

        :param txt: part of text from sentence
        :type txt: string
        :param txt_c: text color
        :type txt_c: string
        :param txt_hglt: color of text highlighting
        :type txt_hglt: string
        :return: string with color labeling
        :rtype: string
        """
        return colored(txt, txt_c, txt_hglt)

    @staticmethod
    def concat_entities(ner_result):
        """
        Concatenation entities from model output on grouped entities

        :param ner_result: output from model pipeline (list of dicts)
        :type ner_result: list
        :return: list of grouped entities with start - end position in text
        :rtype: list
        """
        entities = []
        prev_entity = None
        prev_end = 0
        for i in range(len(ner_result)):

            if (ner_result[i]["entity_group"] == prev_entity) & (
                ner_result[i]["start"] == prev_end
            ):

                entities[i - 1][2] = ner_result[i]["end"]
                prev_entity = ner_result[i]["entity_group"]
                prev_end = ner_result[i]["end"]
            else:
                entities.append(
                    [
                        ner_result[i]["entity_group"],
                        ner_result[i]["start"],
                        ner_result[i]["end"],
                    ]
                )
                prev_entity = ner_result[i]["entity_group"]
                prev_end = ner_result[i]["end"]

        return entities

    def colored_text(self, text: str, entities: list) -> str:
        """
        Highlighting in the text named entities

        :param text: sentence or a part of corpus
        :type text: string
        :param entities: concated entities on groups with start - end position in text
        :type entities: list
        :return: Highlighted sentence
        :rtype: string
        """
        colored_text = ""
        init_pos = 0
        for ent in entities:
            if ent[1] > init_pos:
                colored_text += text[init_pos : ent[1]]
                colored_text += self.text_color(text[ent[1] : ent[2]]) + f"({ent[0]})"
                init_pos = ent[2]
            else:
                colored_text += self.text_color(text[ent[1] : ent[2]]) + f"({ent[0]})"
                init_pos = ent[2]

        return colored_text

    def get_entities(self, text: str):
        """
        Extracting entities from text with them position in text

        :param text: input sentence for preparing
        :type text: string
        :return: list with entities from text
        :rtype: list
        """
        assert len(text) > 0, text
        entities = self.token_pred_pipeline(text)
        concat_ent = self.concat_entities(entities)

        return concat_ent

    def mark_ents_on_text(self, text: str):
        """
        Highlighting named entities in input text

        :param text: input sentence for preparing
        :type text: string
        :return: Highlighting text
        :rtype: string
        """
        assert len(text) > 0, text
        entities = self.get_entities(text)

        return self.colored_text(text, entities)


def print_entities_in_text(entities_in_text: Generator) -> None:
    for sentence in entities_in_text:
        print(sentence)
        print("-*-" * 25)


def safety_mkdir(path):
    extract_dir = os.path.dirname(path)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)


def save_data_to_txt(extracted_entities: Generator, path_to_save: str):
    safety_mkdir(path_to_save)
    with open(path_to_save, "w") as f:
        for i, l_ent in enumerate(extracted_entities):
            for ent in l_ent:
                for_write = [str(i)] + [str(i) for i in ent]
                f.write(" ".join(for_write) + "\n")
    print(f"Result saved at <{path_to_save}>")


@hydra.main(config_path="../config", config_name="infer")
def main(cfg):

    f_path_to_read = os.path.join(cfg.base_path, cfg.path_to_test_file)
    f_path_to_save = os.path.join(cfg.base_path, cfg.path_to_save_data)
    path_to_model = os.path.join(cfg.base_path, cfg.model_checkpoint)

    seqs_example = dl.read_txt_file(f_path_to_read)
    extractor = NerExtractor(model_checkpoint=path_to_model)

    extracted_entities = (extractor.get_entities(i) for i in seqs_example)
    save_data_to_txt(extracted_entities, f_path_to_save)

    # marked_entities_in_text = (extractor.mark_ents_on_text(i) for i in seqs_example)
    # print_entities_in_text(marked_entities_in_text)


if __name__ == "__main__":
    main()
