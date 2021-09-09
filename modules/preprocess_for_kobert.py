# Version 1.4
import re
import numpy as np


# false positive 0개, True Positive 180개, False Negative 2개
def substitute_probability(text):
    if "다항식" in text:
        return text
    if "식을 `" in text:
        return text
    if "방정식" in text:
        return text
    if "점 `" in text or "점을 `" in text:
        return text
    if (
        "점" in text
        and re.findall(r"P\([\w\+\-\(\)/]{1,10},\s?[\w\+\-\(\)/]{1,10}\)", text) != []
    ):
        return text
    else:
        substituted = re.sub("P\s?\(", "확률함수(", text)
        return substituted


def sub_func_replace_pipe(input_text: str) -> str:
    if "|" in input_text and "{" in input_text:
        input_text = re.sub("\|", "조건제시법|", input_text)
    elif "|" in input_text and "확률함수(" in input_text and ")" in input_text:
        words_broken_down = []
        # word_broken_by_bracket = re.findall(r'\([^)]*\)', input_text)
        word_broken_by_bracket = re.findall(
            '\[[^\]]*\]|\([^\)]*\)|"[^"]*"|\S+', input_text
        )
        # print(word_broken_by_bracket)
        for i in word_broken_by_bracket:
            # print(i)
            if i.count("|") % 2 == 0:
                i = re.sub("\|", "절대값|", i)
                words_broken_down.append(i)
            elif i.count("|") % 2 == 1:
                i = re.sub("\|", "조건부확률|", i)
                words_broken_down.append(i)
            else:
                words_broken_down.append(i)
        input_text = "".join(words_broken_down)
    elif input_text.count("|") >= 2:
        input_text = re.sub("\|", "절대값|", input_text)
    else:
        return input_text
    result_text = input_text
    return result_text


#  if list_math_matched.index(math) % 2 == 0
def main_replace_pipe(input_text: str) -> str:
    # print(input_text, "\n")

    list_to_return = []
    # list_separated = re.findall(r'(?<=`).*?(?=`)', input_text)
    # list_separated = input_text.split("`")

    delimiter = "`"
    if input_text.startswith(delimiter) == True:
        bool_starts_with_delimiter = True
    elif input_text.startswith(delimiter) == False:
        bool_starts_with_delimiter = False
    list_separated = [delimiter + e for e in input_text.split(delimiter) if e]

    # print(list_separated, "\n")
    for item in list_separated:
        if "|" in item and "{" in item:
            item = sub_func_replace_pipe(item)
            # item = f"`{item}`"
            list_to_return.append(item)
        elif "|" in item and "확률함수(" in item and ")" in item:
            item = sub_func_replace_pipe(item)
            # item = f"`{item}`"
            list_to_return.append(item)
        elif item.count("|") >= 2:
            item = sub_func_replace_pipe(item)
            # item = f"`{item}`"
            list_to_return.append(item)
        else:
            list_to_return.append(item)
    # print(list_to_return, "\n")
    str_to_return = "".join(list_to_return)
    if bool_starts_with_delimiter == True:
        return str_to_return
    elif bool_starts_with_delimiter == False:
        return str_to_return[1:]


def preprocess(data, korean=True, space=True, condition=True):
    df = data.copy()
    # Math symbols

    # unnecessary tags
    list_html_tags = [
        "</legend>",
        "<br/>",
        "<fieldset>",
        "</span>",
        '<span class="box-text">',
        "</fieldset>",
        "<legend>",
    ]

    list_korean_tags = [
        # "<보 기>",
        "<보기>",
        "<증명>",
        "<규칙>",
        "<그림 `1`>",
        "<그림 `2`>",
        "<단계 `1`>",
        "<단계 `7`>",
        "<조건>",
    ]

    # HANDLING EXCEPTIONS

    # fixing korean tags into normal word
    df["text"] = df["text"].apply(lambda x: re.sub("<보 기>", "보기", x))
    for item in list_korean_tags:
        word_without_bracket = item[1:-2]
        df["text"] = df["text"].apply(lambda x: re.sub(item, word_without_bracket, x))

    # remove html tags from dataset
    for item in list_html_tags:
        df["text"] = df["text"].apply(lambda x: re.sub(item, " ", x))

    # remove 보기 ㄱ. ㄴ. ...
    df["text"] = df["text"].apply(lambda x: re.sub(r"[ㄱ-ㅎ]\.", "", x))
    # remove (가), (나), ...
    df["text"] = df["text"].apply(lambda x: re.sub(r"\([가-힣]\)", "", x))

    # handle specified math terms
    df["text"] = df["text"].apply(lambda x: re.sub("\{::\}", " ", x))  # Remove {::}
    df["text"] = df["text"].apply(lambda x: re.sub("!=", "≠", x))
    df["text"] = df["text"].apply(lambda x: re.sub("rarr|->", "→", x))
    df["text"] = df["text"].apply(lambda x: re.sub("nn", "∩", x))
    df["text"] = df["text"].apply(lambda x: re.sub("uu", "∪", x))
    df["text"] = df["text"].apply(lambda x: re.sub("sub", "⊂", x))
    df["text"] = df["text"].apply(lambda x: re.sub("sup", "⊃", x))
    df["text"] = df["text"].apply(lambda x: re.sub("[^s\w\n]notin[\w\n]", "∉", x))
    df["text"] = df["text"].apply(lambda x: re.sub("[^s\w\n]in[\w\n]", "∈", x))
    df["text"] = df["text"].apply(lambda x: re.sub("emptyset", "Ø", x))
    df["text"] = df["text"].apply(lambda x: re.sub("cdots", "…", x))
    df["text"] = df["text"].apply(lambda x: re.sub("<=", "≤", x))
    df["text"] = df["text"].apply(lambda x: re.sub(">=", "≥", x))
    df["text"] = df["text"].apply(lambda x: re.sub("xx", "*", x))
    df["text"] = df["text"].apply(lambda x: re.sub("sqrt", "√", x))
    df["text"] = df["text"].apply(lambda x: re.sub("root", "√", x))
    df["text"] = df["text"].apply(lambda x: re.sub("[^r]oo", "∞", x))
    df["text"] = df["text"].apply(lambda x: re.sub("/_", "∠", x))
    df["text"] = df["text"].apply(lambda x: re.sub("sum_", "∑", x))
    # _(2n)C_0나 _nC_k, _nC_1 등을 조합으로 바꿔줘야 한다.
    df["text"] = df["text"].apply(lambda x: re.sub("C_", "조합", x))
    df["text"] = df["text"].apply(lambda x: re.sub("P_", "순열", x))

    if korean == True:
        # df["text"] = df["text"].apply(
        #    lambda x: re.sub("[P]\(\w{1,2}?\, \w{1,2}?\)", "분할수", x))
        # df["text"] = df["text"].apply(lambda x: re.sub("\^2", "제곱", x))
        # 제곱수를 kobert tokenizer가 인식하지 못해 한글로 치환
        # 다항식 P(x)나 P(1,2) 등을 확률함수로 바꾸는 경우 해결
        df["text"] = df["text"].apply(lambda x: substitute_probability(x))
        # 절대값
        df["text"] = df["text"].apply(lambda x: main_replace_pipe(x))
        # "점"이랑 "변"은 Expectation에 의외로 같이 많이 섞여있음.
        df["text"] = df["text"].apply(
            lambda x: re.sub("E\(", "기댓값(", x) if not "기울기" in x else x
        )  # 0개 잘못 분류
        df["text"] = df["text"].apply(
            lambda x: re.sub("V\(", "분산(", x)
        )  # 분류 잘못 하는 거 없음
        df["text"] = df["text"].apply(lambda x: re.sub("\^C", "여집합", x))
        df["text"] = df["text"].apply(lambda x: re.sub("분포 \`[A-Z]\(.+?`", "분포", x))

    # Add spaces to the next of math terms
    if space == True:
        math_terms = [
            "sin",  # Triangle function
            "cos",
            "tan",
            "alpha",  # Roman letters
            "beta",
            "theta",
            "gamma",
            "omega",
            "phi",
            "uu",  # Sets
            "nn",
            "sup",
            "sub",
            "^C",
            "pi",  # Etc
            "abs",
            "sqrt",
            "cdots",
            "bar",
        ]

        math_symbols = [
            "!",
            "\(",
            "\)",
            "\{",
            "\}",
            "\[",
            "\]",
            "\+",
            "\-",
            "\|",
            "/_",
            "@",
            "=",
        ]

        for math in math_terms:
            df["text"] = df["text"].apply(
                lambda x: re.sub(fr"([^\s])({math})", fr"\1 \2", x)
            )
            df["text"] = df["text"].apply(
                lambda x: re.sub(fr"({math})([^\s])", fr"\1 \2", x)
            )
            # df["text"] = df["text"].apply(lambda x: x.replace(math, " " + math + " "))

    if condition == True:
        df["text"] = df["text"].apply(lambda x: re.sub("\(단,.+?\)", "", x))

    return df


def preprocess_noisy(data, korean=True, space=True, condition=True):

    df = data.copy()
    df["text"] = df["text"].apply(lambda x: re.sub("\\r|\\n", "", x))
    df["text"] = df["text"].apply(lambda x: re.sub("left|right", "", x))

    df["text"] = df["text"].apply(lambda x: re.sub("leq", "<=", x))
    df["text"] = df["text"].apply(lambda x: re.sub("le", "<", x))
    df["text"] = df["text"].apply(lambda x: re.sub("ge", ">", x))
    df["text"] = df["text"].apply(lambda x: re.sub("geq", ">=", x))
    df["text"] = df["text"].apply(lambda x: re.sub("times", "xx", x))

    df["text"] = df["text"].apply(
        lambda x: re.sub(r"\\dfrac\s+?\{(.+?)\}\s+?\{(.+?)\}", r"\1/\2", x)
    )
    df["text"] = df["text"].apply(lambda x: re.sub("\{|\}", "", x))

    df["text"] = df["text"].apply(lambda x: re.sub(r"\\", "", x))
    df["text"] = df["text"].apply(lambda x: re.sub(r"\$", "", x))
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s\s+", " ", x))

    df = preprocess(df, korean=korean, space=space, condition=condition)
    return df


# Drop useless data in training dataset
# ONLY FOR TRAIN DATASET
def drop_noise(data):

    df = data.copy()

    noise_index = []
    noise_index += list(df[df["text"] == "다음 중 옳은 것은?"].index)
    noise_index += list(df[df["text"] == "다음 중 옳지 않은 것은?"].index)
    noise_index += list(df[df["text"] == "다음 설명 중 옳은 것은?"].index)

    df.drop(index=noise_index, inplace=True)
    return df
