import os
import sys
from time import strftime
import logging
PATH = os.path.abspath('..') + '/logs/'

if not os.path.exists(PATH):
    os.makedirs(PATH)
    print(f"The directory {PATH} has been created.")
else:
    print(f"The directory {PATH} already exists.")

FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'


class MyLog(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MyLog, cls).__new__(cls)
            cls._instance.logger = logging.getLogger()
            cls._instance.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
            current_time = strftime("%Y-%m-%d_%H-%M")
            cls._instance.log_filename = f"{PATH}{current_time}.log"
            cls._instance.logger.addHandler(cls._instance.get_file_handler(cls._instance.log_filename))
            cls._instance.logger.addHandler(cls._instance.get_console_handler())
            cls._instance.logger.setLevel(logging.DEBUG)
        return cls._instance

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    # 输出到控制台handler的函数定义
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    # 获取日志文件路径的方法
    def get_log_file_path(self):
        return self.log_filename
