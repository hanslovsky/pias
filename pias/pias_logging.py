import logging
print(logging)

trace = logging.DEBUG- 5
logging.TRACE = trace
logging.addLevelName(trace, 'TRACE')

class PiasLogger(logging.getLoggerClass()):
    def trace(self, msg, *args, **kwargs):
        self.log(trace, msg, *args, **kwargs)
logging.setLoggerClass(PiasLogger)

levels = ('NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL', 'FATAL', 'TRACE')