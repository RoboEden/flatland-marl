import sys
from subprocess import Popen, CalledProcessError

import ffmpeg
import numpy as np


class VideoWriter:
    def __init__(self, filename, fps=30, width=None, height=None) -> None:
        """A video writer using ffmpeg. Example usage:

        ```python
        import numpy as np

        video = VideoWriter("output.mp4", fps=30)

        width, height = 400, 300
        for i in range(256):
            frame = i*np.ones((height, width, 3), dtype=np.uint8)
            video.write_frame(frame)

        video.close()
        ```

        Args:
            filename (str): output filename.
            fps (int, optional): frame per second. Defaults to 30.
            width (int, optional): video width. If not given, decided by frames written.
            height (int, optional): video height. If not given, decided by frames written.
        """
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self.ffmpeg_process: Popen = None

    def write(self, frame: np.ndarray):
        """Write a frame to the video.

        Args:
            frame (np.ndarray): np.ndarray in size (height, width, 3) with dtype np.uint8.
        """
        frame = self._pad_frame(frame)
        self._update_info(frame)
        try:
            self.ffmpeg_process.stdin.write(frame.tobytes())
        except:
            self.check_ffmpeg_returncode()

    def close(self):
        """Finish writing the video.

        Raises:
            ValueError: If no frame has been written, raise ValueError.
            RuntimeError: If ffmpeg fails.
        """
        if self.ffmpeg_process is None:
            raise ValueError("No frame is written.")
        self.ffmpeg_process.stdin.close()
        self.check_ffmpeg_returncode()

    def print_ffmpeg_output(self):
        """Stream ffmpeg's stdout to stdout and its stderr to stderr."""
        sys.stdout.write(self.ffmpeg_process.stdout.read().decode())
        sys.stderr.write(self.ffmpeg_process.stderr.read().decode())

    def check_ffmpeg_returncode(self):
        """Raise CalledProcessError if the exit code is non-zero."""
        ffmpeg_process = self.ffmpeg_process
        ffmpeg_process.wait()
        if ffmpeg_process.returncode:
            self.print_ffmpeg_output()
            print()
            raise CalledProcessError(
                ffmpeg_process.returncode,
                ffmpeg_process.args,
                ffmpeg_process.stdout,
                ffmpeg_process.stderr,
            )

    def _pad_frame(self, frame):
        height, width = frame.shape[:2]
        if height % 2 == 1:
            frame = np.concatenate(
                [frame, np.zeros((1, width, 3), dtype=np.uint8)], axis=0
            )
            height += 1
        if width % 2 == 1:
            frame = np.concatenate(
                [frame, np.zeros((height, 1, 3), dtype=np.uint8)], axis=1
            )
            width += 1
        return frame

    def _update_info(self, frame: np.ndarray):
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # three chanel
        height, width = frame.shape[:2]

        if self.width is None:
            self.width = width
        else:
            assert self.width == width

        if self.height is None:
            self.height = height
        else:
            assert self.height == height

        if self.ffmpeg_process is None:
            self.ffmpeg_process = self._create_ffmpeg_process()

    def _create_ffmpeg_process(self):
        assert self.height is not None
        assert self.width is not None
        ffmpeg_process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(self.width, self.height),
                r=self.fps,
            )
            .output(self.filename, pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        return ffmpeg_process


if __name__ == "__main__":
    width, height = 401, 301

    video = VideoWriter("output.mp4", 30)

    for i in range(256):
        frame = i * np.ones((height, width, 3), dtype=np.uint8)
        video.write(frame)

    video.close()
