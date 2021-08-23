import cv2
import time
import threading


class VideoFileReader:
    """
    A video reader that internally reads the frames in a separated thread at the right fps,
    and returns the latest (current) frame when read is called.
    """

    def __init__(self):
        self.file_name = None
        self.cap = None
        self.fps = None

        self.thread = threading.Thread(target=self.__thread_function)
        self.lock = threading.Lock()
        self.last_frame = None
        self.last_ret = None
        self.running = False

    # ------------------------------------------------------------------------------------------------

    def __thread_function(self):
        last_timestamp = None
        frame_interval = 1.0 / self.fps

        while self.running:
            timestamp = time.time()

            # read next frame if sufficient time has elapsed since the last capture
            if last_timestamp is None or timestamp - last_timestamp > frame_interval:
                ret, frame = self.cap.read()
                last_timestamp = time.time()

                with self.lock:
                    self.last_frame = frame
                    self.last_ret = ret

                    if not ret:
                        self.running = False

            sleep_duration = max(0, frame_interval - (time.time() - last_timestamp))
            time.sleep(sleep_duration)

    # ------------------------------------------------------------------------------------------------

    def start(self, file_name, start_frame=None, custom_fps=None):
        self.cap = cv2.VideoCapture(file_name)
        assert (self.cap.isOpened())
        if custom_fps:
            self.fps = custom_fps
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.running = True
        if start_frame is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        self.thread.start()

    # ------------------------------------------------------------------------------------------------

    def stop(self):
        self.cap = None
        self.running = False
        self.thread.join()

    # ------------------------------------------------------------------------------------------------

    def restart(self):
        assert self.file_name

        self.stop()
        self.start(self.file_name)

    # ------------------------------------------------------------------------------------------------

    def read(self):
        # read the last frame obtained by the thread
        n_tries = 0
        max_retries = 50

        while True:
            n_tries += 1
            with self.lock:
                if self.last_frame is not None:
                    frame = self.last_frame
                    self.last_frame = None
                    ret = self.last_ret
                    self.last_ret = None
                    return ret, frame

            if n_tries > max_retries:
                return False, None

            # the frame was not ready, wait and then retry
            time.sleep(0.001)


