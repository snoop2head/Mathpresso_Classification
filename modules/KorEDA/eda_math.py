import random
import pickle
import re
import pathlib
import pandas as pd
import numpy as np


dict_more_than_200Q = {
    # "H1S2-04" : 394,
    "H1S2-05": 328,
    "H1S2-03": 247,
    "H1S2-06": 242,
}


dict_less_than_200Q = {
    "HSU1-06": 198,
    "H1S1-08": 189,
    "H1S1-07": 165,
    "HSU1-03": 164,
    "HSU1-04": 163,
    "H1S1-11": 157,
    "HSU1-02": 133,
    "H1S1-10": 133,
    "HSTA-03": 121,
}


dict_less_than_150Q = {
    "HSU1-07": 116,
    "HSTA-06": 111,
    "H1S2-02": 111,
    "H1S1-05": 110,
    "HSU1-10": 110,
    "H1S1-02": 108,
    "HSU1-01": 107,
    "HSTA-07": 103,
    "HSU1-08": 101,
    "H1S1-12": 98,
    "HSTA-05": 97,
    "HSU1-09": 97,
}

# chapters with less than 100 questions
dict_less_than_100Q = {
    "HSU1-05": 93,
    "H1S1-09": 92,
    "HSTA-02": 92,
    "HSU1-11": 90,
    "HSTA-01": 88,
    "H1S1-03": 85,
    "HSTA-04": 84,
    "H1S1-06": 78,
    "H1S1-04": 69,
    "H1S1-01": 64,
    "H1S2-01": 60,
}


# 2 배수를 곱해서 500문제 이상으로 맞춤
list_more_than_200Q = list(dict_more_than_200Q.keys())

# 4 배수를 곱해서 500문제이상으로 맞춤
list_less_than_200Q = list(dict_less_than_200Q.keys())

# 5배수를 곱해서 500문제이상으로 맞춤
list_less_than_150Q = list(dict_less_than_150Q.keys())

# 6배수 이상을 곱해서 400문제 언저리로 맞춤
list_less_than_100Q = list(dict_less_than_100Q.keys())


# sample_text = "다항식 `ax^3-13x^2+bx-3`를 `(x-2)^2`으로 나누었을 때의 나머지가 `x+1`이 되도록 하는 상수 `a`, `b`에 대하여 `a+b`의 값은?"

# Random Insertion from the chapter corpus
# Random letter-based Deletion: letters inside of `` or letter inside of korean words
# Random word-based Deletion: word inside of `` or word inside of korean words
# Random Order Rearrangement
# 하나씩만 해보면서 성능향상이 있는지 측정

# 한글만 남기고 나머지는 삭제
def get_only_hangul(line):
    dict_order = {}
    list_parseText = re.compile("ㄱ-ㅎ|ㅏ-ㅣ|[가-힣]+").findall(line)
    # print(list_parseText)
    # parseText = " ".join(list_parseText)
    # print(parseText)
    for word in list_parseText:
        # print(word)
        # a = re.search(r"\b(laugh)\b", string)
        # print(a)
        index_no = line.find(word)
        # print(index_no)
        dict_order[f"{word}"] = index_no
    return dict_order


# print(get_only_hangul(sample_text))

# math notation들을 dictionary에 저장
def get_math_notations(line):
    dict_order = {}
    list_math_matched = re.findall(r"(?<=`).*?(?=`)", line)
    list_math = [
        f"`{math}`"
        for math in list_math_matched
        if list_math_matched.index(math) % 2 == 0
    ]

    """
    list_hangul = [
        f"{word}"
        for word in list_math_matched
        if list_math_matched.index(word) % 2 == 1
    ]

    print(list_math, list_hangul)
    """
    # print(list_math)
    for word in list_math:
        index_no = line.find(word)
        dict_order[f"{word}"] = index_no
    return dict_order


def get_sorted_dict(line):
    x = get_only_hangul(line)
    y = get_math_notations(line)
    z = {**x, **y}
    # print(z)

    sorted_z = dict(sorted(z.items(), key=lambda item: item[1]))
    return sorted_z


def get_sorted_list(dictionary):
    sorted_list = list(dictionary.keys())
    return sorted_list


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(list_words, p):
    if len(list_words) == 1:
        return list_words

    list_new_words = []
    for word in list_words:
        r = random.uniform(0, 1)
        if r > p:
            list_new_words.append(word)

    if len(list_new_words) == 0:
        rand_int = random.randint(0, len(list_words) - 1)
        return [list_words[rand_int]]

    return list_new_words


"""
print(get_sorted_list(get_sorted_dict(sample_text)))
print(random_deletion(get_sorted_list(get_sorted_dict(sample_text)), p=0.1))
"""


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)

    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words


"""
alpha_rs = 0.1
num_words = len(get_sorted_list(get_sorted_dict(sample_text)))

print(get_sorted_list(get_sorted_dict(sample_text)))
print(
    random_swap(
        get_sorted_list(get_sorted_dict(sample_text)), max(1, int(alpha_rs * num_words))
    )
)
"""


def eda_math(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    words = get_sorted_list(get_sorted_dict(sentence))
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 2) + 1

    # n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    """
    # SR: Synonym Replacement, 특정 단어를 유의어로 교체
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(" ".join(a_words))
        print("SR: ", " ".join(a_words))
    

    # RI: Random Insertion, 임의의 단어를 삽입
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(" ".join(a_words))
        print("RI: ", " ".join(a_words))
    """

    # RS: Random Swap, 문장 내 임의의 두 단어의 위치를 바꿈
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))
        # print("RS: ", " ".join(a_words))

    # RD: Random Deletion: 임의의 단어를 삭제
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(" ".join(a_words))
        # print("RD: ", " ".join(a_words))

    random.shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
        ]

    augmented_sentences.append(sentence)

    set_augmented_sentences = set(augmented_sentences)
    list_augmented_sentences = list(set_augmented_sentences)
    # print(list_augmented_sentences[0])

    return list_augmented_sentences


# print(EDA(sample_text))


def eda_df(df_train, label_name="qtid", even=True):
    df_sizeup = df_train.copy()
    for index, row in df_train.iterrows():
        str_question = row["text"]
        list_sentences = eda_math(
            str_question, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=10
        )

        if label_name == "qtid":
            chapter_label = row["qtid"][:7]
        elif label_name == "chapter":
            chapter_label = row["chapter"]
        else:
            print("중단원 기준인 chapter을 입력하거나, 소단원 기준인 qtid를 입력하세요.")

        if next(
            (
                str_chapter
                for str_chapter in list_more_than_200Q
                if chapter_label in str_chapter
            ),
            False,
        ):
            list_EDA = random.choices(list_sentences, k=2)
        elif next(
            (
                str_chapter
                for str_chapter in list_more_than_200Q
                if chapter_label in str_chapter
            ),
            False,
        ):
            if len(list_sentences) >= 4:
                list_EDA = random.choices(list_sentences, k=4)
            else:
                list_EDA = list_sentences
        elif next(
            (
                str_chapter
                for str_chapter in list_less_than_150Q
                if chapter_label in str_chapter
            ),
            False,
        ):
            if len(list_sentences) >= 5:
                list_EDA = random.choices(list_sentences, k=5)
            else:
                list_EDA = list_sentences
        elif next(
            (
                str_chapter
                for str_chapter in list_less_than_100Q
                if chapter_label in str_chapter
            ),
            False,
        ):
            list_EDA = list_sentences
        else:
            list_EDA = []
        for item in list_EDA:
            row_created = row.copy()
            row_created["text"] = item
            df_sizeup = df_sizeup.append(row_created, ignore_index=True)
    return df_sizeup
