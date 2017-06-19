from __future__ import print_function, absolute_import
import sys
import logging
from tqdm import tqdm


class ProgressBar(tqdm):
    """
    Implements a simple progress bar using TQDM

    Examples:
        progress_bar = ProgressBar(unit='batches', ncols=100, total=nbatches)
        for sample in progress_bar(dataset):
            result = fun(sample)
    """
    def __call__(self, iterable):
        self.iterable = iterable
        return self


class PBStreamHandler(logging.Handler):
    """
    Implements a StreamHandler for the python logging module that plays nicely with ProgressBar

    Examples:
        logger = logging.getLogger()
        logger.addHandler(PBStreamHandler(logging.INFO))
        logger.setLevel(logging.INFO)

        progress_bar = ProgressBar()
        for sample in progress_bar(dataset):
            result = fun(sample)
            logger.info("Result: {}".format(result))

    Modified from:
    http://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
    """
    def __init__(self, stream=sys.stdout, level=logging.NOTSET):
        self.stream = stream
        super(PBStreamHandler, self).__init__(level=level)

    def emit(self, record):
        try:
            msg = self.format(record)
            ProgressBar.write(msg, file=self.stream)
            self.flush()
        except(KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
