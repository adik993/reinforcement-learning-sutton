import logging

import sys

formatter = logging.Formatter('%(asctime)-15s - %(levelname)-5s - %(message)s')


def make_logger(name, level=logging.DEBUG):
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log = logging.getLogger(name)
    log.setLevel(level)
    log.addHandler(console)
    log.propagate = False
    return log


def make_file_logger(name, filename, level=logging.DEBUG):
    file = logging.FileHandler(filename, mode='w')
    file.setFormatter(formatter)
    log = logging.getLogger(name)
    log.setLevel(level)
    log.addHandler(file)
    log.propagate = False
    return log
