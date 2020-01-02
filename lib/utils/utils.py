import numpy as np

"""
some really utils functions
"""


def get_score_label_array_from_dict(score_dict, label_dict):
    """
    :param score_dict: defaultdict(list)
    :param label_dict: defaultdict(list)
    :return: np array with score and label
    """
    assert len(score_dict) == len(label_dict), "The score_dict and label_dict don't match"
    score = np.ones(len(score_dict))
    label = np.ones(len(label_dict))

    for idx, (key, score_l) in enumerate(score_dict.items()):
        label[idx] = max(label_dict[key])
        score[idx] = max(score_l)
    return score, label