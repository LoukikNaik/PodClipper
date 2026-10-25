"""Run transcript search and review alongside one reel export at a time."""

from contextlib import contextmanager
from queue import SimpleQueue
from threading import Event, Thread


@contextmanager
def prefetch_selection(selections):
    queue, stopped = SimpleQueue(), Event()

    def produce():
        try:
            while not stopped.is_set():
                try:
                    clip = next(selections)
                except StopIteration:
                    break
                if not stopped.is_set():
                    queue.put(('clip', clip))
        except Exception as exc:
            queue.put(('error', exc))
        finally:
            queue.put(('done', None))
            close = getattr(selections, 'close', None)
            if close is not None:
                close()

    def consume():
        while True:
            kind, value = queue.get()
            if kind == 'done':
                return
            if kind == 'error':
                raise value
            yield value

    thread = Thread(target=produce, name='podclipper-selection', daemon=True)
    thread.start()
    try:
        yield consume()
    finally:
        stopped.set()
        thread.join(timeout=1)
