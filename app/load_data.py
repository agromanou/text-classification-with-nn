import os
import re
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup as Soup
from bs4 import Tag

from app import DATA_DIR

TRAIN_FILE = 'ABSA16_Laptops_Train_SB1_v2.xml'
TEST_FILE = 'EN_LAPT_SB1_TEST_.xml.gold'


def calculate_label_ratio(labels):
    """
    :param labels:
    :return:
    """

    x = Counter(labels)
    sum_counts = sum(x.values())

    print()
    for t in x.most_common():
        ratio = round(t[1] / sum_counts * 100, 2)

        print('Label: {}, Instances: {}, Ratio: {}%'.format(t[0], t[1], ratio))


def parse_reviews(file_type='train',
                  save_data=True,
                  load_data=True):
    """
    :param file_type: str. the file type. Enum between 'train' and 'test'
    :param save_data: bool. whether the extracted data will be saved
    :param load_data: bool. whether the extracted data should be loaded
    :return: pandas data-frame with 2 columns: polarity, text
    """
    assert file_type in ['train', 'test']

    file = TRAIN_FILE if file_type == 'train' else TEST_FILE
    path = os.path.join(DATA_DIR, file)

    if load_data:
        try:
            x = path.split('.')[-1]
            infile = re.sub(x, 'csv', path)
            print('Loading file: {}'.format(infile))

            return pd.read_csv(infile)
        except FileNotFoundError:
            print('File Not Found on Data Directory. Creating a new one from scratch')

    data = list()

    handler = open(path).read()
    soup = Soup(handler, "lxml")

    reviews = soup.body.reviews

    for body_child in reviews:
        if isinstance(body_child, Tag):
            for body_child_2 in body_child.sentences:
                if isinstance(body_child_2, Tag):
                    if body_child_2.opinions:
                        opinion = body_child_2.opinions.opinion
                        # keeping only reviews that have a polarity
                        if opinion:
                            sentence = body_child_2.text.strip()
                            polarity = opinion.attrs.get('polarity')
                            data.append({'text': sentence, 'polarity': polarity})

    extracted_data = pd.DataFrame(data)
    extracted_data = extracted_data[extracted_data['polarity'] != 'neutral']
    if save_data:
        print('Saving etracted reviews metadata from file: {}'.format(file_type))
        x = path.split('.')[-1]
        outfile = re.sub(x, 'csv', path)
        extracted_data.to_csv(outfile, encoding='utf-8', index=False)

    return extracted_data


if __name__ == "__main__":
    train_data = parse_reviews(load_data=False, save_data=False, file_type='train')
    print(train_data.head())

    calculate_label_ratio(train_data['polarity'])
