from threading import Thread, Lock
import time


# ------------------------------------------------------------------------------------------------

class AsyncCaller:
    """
    Calls a given function, allowing to check for
    completion and to retrieve the result asynchronously.
    """

    def __init__(self, fn):
        self._fn = fn
        self._args = None
        self._result = None
        self._lock = Lock()
        self._thread = Thread(target=self._thread_function)


# ------------------------------------------------------------------------------------------------

    def _thread_function(self):
        result = self._fn(*self._args)

        self._lock.acquire()
        self._result = result
        self._lock.release()


# ------------------------------------------------------------------------------------------------

    def call(self, *args):
        self._lock.acquire()
        if self._thread.is_alive():
            assert False, 'multiple simultaneous calls are not allowed'
        self._result = None
        self._args = args
        self._lock.release()

        self._thread.start()


# ------------------------------------------------------------------------------------------------

    def get_result(self):
        self._lock.acquire()
        result = self._result
        self._lock.release()

        return result


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    def f(t1, t2):
        print('started processing', t1, t2)
        time.sleep(t1 + t2)
        print('done processing')
        return 123

    async_f = AsyncCaller(f)
    async_f.call(2, 3)
    while True:
        time.sleep(0.3)
        r = async_f.get_result()
        if r is None:
            print('result not ready')
        else:
            print(r)
            break


# ------------------------------------------------------------------------------------------------
