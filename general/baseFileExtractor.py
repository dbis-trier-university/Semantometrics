def get_seminal_s():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_to_seminal_stemmed.json="):
            return line.split("path_to_seminal_stemmed.json=")[1][:-1]


def get_survey_s():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_to_survey_stemmed.json="):
            return line.split("path_to_survey_stemmed.json=")[1][:-1]


def get_uninfluential_s():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_to_uninfluential_stemmed.json="):
            return line.split("path_to_uninfluential_stemmed.json=")[1][:-1]


def get_seminal_u():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_to_seminal_unstemmed.json="):
            return line.split("path_to_seminal_unstemmed.json=")[1][:-1]


def get_survey_u():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_to_survey_unstemmed.json="):
            return line.split("path_to_survey_unstemmed.json=")[1][:-1]


def get_uninfluential_u():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_to_uninfluential_unstemmed.json="):
            return line.split("path_to_uninfluential_unstemmed.json=")[1][:-1]


# ----------------------------------------------------------------------------------------------------------------------


def get_file_base():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_for_everything="):
            return line.split("path_for_everything=")[1][:-1]


def get_stem():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("stem="):
            return line.split("stem=")[1][:-1] == 'True'


# ----------------------------------------------------------------------------------------------------------------------


def get_d2v_base():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_d2v_base="):
            return line.split("path_d2v_base=")[1][:-1]


def get_lda_base():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("path_lda_base.txt="):
            return line.split("path_lda_base.txt=")[1][:-1]


# ----------------------------------------------------------------------------------------------------------------------


def get_what_to_do():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("what_to_do="):
            return int(line.split("what_to_do=")[1][:-1])


def get_which_vectors():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("vectors="):
            return line.split("vectors=")[1][:-1]


def get_which_distance():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("distance_measure="):
            return line.split("distance_measure=")[1][:-1]


def get_classifier():
    f = open("../specification.txt", "r")
    while True:
        line = f.readline()
        if line.startswith("classifier="):
            return line.split("classifier=")[1][:-1]
