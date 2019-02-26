import logging
import threading

# see https://stackoverflow.com/a/24796823/1725687

class CountDownLatch(object):
    def __init__(self, count=1):
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.count = count
        self.lock = threading.Condition()

    def count_down(self):
        with self.lock:
            self.logger.debug('Decrementing %d by one', self.count)
            self.count -= 1
            if self.count <= 0:
                self.lock.notifyAll()

    def get_count(self):
        with self.lock:
            return self.count

    def wait_for_countdown(self, timeout=None):
        with self.lock:
            if self.count > 0:
                self.lock.wait(timeout=timeout)

    def wait_until_value(self, value=0, timeout=None):
        '''

        :param value:
        :param timeout: in seconds
        :return:
        '''
        assert value >= 0, 'Received illegal value %d < 0 -- would never return' % value
        with self.lock:
            while self.count > value:
                self.lock.wait(timeout=timeout)