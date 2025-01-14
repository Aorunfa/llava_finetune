import logging
class Logger(object):
    def __init__(self, path=None, log_name='log', mode='a'):
        if path is None:
            self.log_path = 'action_log.log'
        else:
            self.log_path = path
        self.mode = mode
        self.log_name = log_name
        self.logger = logging.getLogger(self.log_name)
        self.set_up()

    def set_up(self):
        """
        日志设置初始化
        :return:
        """
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(filename=self.log_path,
                                           encoding='utf-8',
                                           mode=self.mode)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt='%(asctime)s : %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        # 文件保存
        file_handler.setFormatter(fmt=formatter)
        file_handler.setLevel(level=logging.INFO)
        # 工作台打印
        stream_handler.setFormatter(fmt=formatter)
        stream_handler.setLevel(level=logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def __call__(self, log_info):
        self.logger.info(log_info)