import threading

# see: https://stackoverflow.com/a/48433648/1725687

class AtomicInteger():
    def __init__(self, value=0):
        self._value = value
        self._lock = threading.Lock()

    def increment_and_get(self):
        with self._lock:
            self._value += 1
            return self._value

    def deccrement_and_get(self):
        with self._lock:
            self._value -= 1
            return self._value

    def get_and_increment(self):
        with self._lock:
            value = self._value
            self._value += 1
        return value

    def get_and_decrement(self):
        with self._lock:
            value = self._value
            self._value -= 1
        return value

    @property
    def value(self):
        with self._lock:
            return self._value