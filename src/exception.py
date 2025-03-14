import sys
from src.logger import logging

def error_message_detail(error : Exception, error_detail : sys) -> str:
    _,_,exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f'Error occurred in the script {file_name} at line {line_number} : {str(error)}'

class CustomException(Exception):
    def __init__(self , error_message : Exception , error_detail : sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message , error_detail)
        logging.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
