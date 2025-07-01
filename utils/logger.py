import logging

from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """
    Helper class for logging.
    Write the log into a file and a 
    Arguments:
        log_dir (str): Path to log file.
    """
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        event_write_file_name = self.writer.file_writer.event_writer._file_name
        self.logger = self._get_logger(log_dir, event_write_file_name)
        self.log_dir = log_dir

        print (f'Logging to file: {event_write_file_name}')
        
    def _get_logger(self, log_dir, event_write_file_name):
        logger = logging.getLogger(log_dir)
        
        # set up the logger
        logger.setLevel(logging.DEBUG)
        
        # Create a console output handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a file output handler
        file_handler = logging.FileHandler(event_write_file_name + ".log")
        file_handler.setLevel(logging.DEBUG)

        # Define the log format
        format_str = '[%(asctime)s] %(message)s'
        # format_str = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

    def write(self, step, message, tb_dict):
        for key, value in tb_dict.items():
            self.writer.add_scalar(key, value, step)
        if message is not None:
            self.logger.info(message)
