from enum import Enum
from typing import List, Literal, Tuple
import cv2 as cv
from PIL import Image
import numpy as np

# Shot change at 1178 ->1179
class FrameUtil:
    @staticmethod
    def compare_histogram(
        frame1: np.ndarray,
        frame2: np.ndarray,
        compare_method: int = cv.HISTCMP_BHATTACHARYYA,
    ) -> float:

        # Convert it to HSV
        hsv_current_frame = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
        hsv_next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)

        # Calculate the histogram and normalize it
        hist_img1 = cv.calcHist(
            [hsv_current_frame], [0, 1], None, [180, 256], [0, 180, 0, 256]
        )
        cv.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        hist_img2 = cv.calcHist(
            [hsv_next_frame], [0, 1], None, [180, 256], [0, 180, 0, 256]
        )
        cv.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        return cv.compareHist(hist_img1, hist_img2, compare_method)

    histogram_compare_methods = {cv.HISTCMP_BHATTACHARYYA: "Bhattacharyya distance"}

    @staticmethod
    def get_histogram_comparison_method_name(compare_method: int):
        return True
        # return histogram_compare_methods[compare_method]


class RGBVideo:
    def __init__(self, file_name: str, width=480, height=270, channels=3):
        self.file_name = file_name
        self.width = width
        self.height = height
        self.channels = channels

        self.frames_buffer = np.fromfile(file_name, dtype=np.uint8)

        # Calculate number of frames by dividing number of values by size of 1 frame
        self.number_of_frames = int(
            self.frames_buffer.shape[0] / (self.width * self.height * self.channels)
        )

        self.frames_buffer = self.frames_buffer.reshape(
            (self.number_of_frames, self.channels, self.height, self.width)
        )

        # Shape = (9000, 3, 270, 480)

        # Swap Channel and Width to get (9000, 270, 480, 3)
        self.frames_buffer = np.swapaxes(self.frames_buffer, 1, 3)

        # Swap Width and Height to get (9000, 480, 270, 3)
        self.frames_buffer = np.swapaxes(self.frames_buffer, 1, 2)

        print(
            f"Loaded '{self.file_name}' - {self.number_of_frames} frames loaded into buffer"
        )

    def dump_frames(self, output_folder="video_frames"):

        print(f"Dumping frames into '{output_folder}' folder")

        for frame_index, frame_buffer in enumerate(self.frames_buffer):
            im = Image.fromarray(frame_buffer)
            im.save(f"{output_folder}/{frame_index}.png")

    def detect_shots(
        self,
        compare_method: int = cv.HISTCMP_BHATTACHARYYA,
        compare_threshold: float = 0.7,
    ):
        self.shots: List[Tuple[int, int]] = []

        shot_start_frame_index = 0

        for frame_index in range(0, len(self.frames_buffer) - 1):
            current_frame: np.ndarray = self.frames_buffer[frame_index]
            next_frame: np.ndarray = self.frames_buffer[frame_index + 1]

            compare_val = FrameUtil.compare_histogram(
                current_frame, next_frame, compare_method
            )

            if compare_val > compare_threshold:
                self.shots.append((shot_start_frame_index, frame_index))
                shot_start_frame_index = frame_index

                print(f"Frame {frame_index} -> {frame_index + 1} = {compare_val}")

        self.shots.append((shot_start_frame_index, len(self.frames_buffer) - 1))

        print(self.shots)


def main():
    rgb_video = RGBVideo("dataset-001/dataset/Videos/data_test1.rgb")
    rgb_video.detect_shots()


if __name__ == "__main__":
    main()
