import os
from io import BytesIO
from typing import List, Set, Tuple

import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from google.cloud import vision
from PIL import Image

load_dotenv()

client = vision.ImageAnnotatorClient.from_service_account_json(
    os.environ["GOOGLE_VISION_CREDS_JSON"]
)

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

    histogram_compare_methods = {
        cv.HISTCMP_CORREL: "Correlation",
        cv.HISTCMP_CHISQR: "Chi-Square",
        cv.HISTCMP_INTERSECT: "Intersection",
        cv.HISTCMP_BHATTACHARYYA: "Bhattacharyya distance",
        # cv.HISTCMP_HELLINGER: "Bhattacharyya distance",
        cv.HISTCMP_CHISQR_ALT: "Alternative Chi-Square",
        cv.HISTCMP_KL_DIV: "Kullback-Leibler divergence",
    }

    @classmethod
    def get_histogram_comparison_method_name(cls, compare_method: int):
        return cls.histogram_compare_methods[compare_method]


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

        # Initialize shots as the whole video
        self.shots: List[Tuple[int, int]] = [(0, self.number_of_frames - 1)]

        # Shape = (9000, 3, 270, 480)
        self.frames_buffer = self.frames_buffer.reshape(
            (self.number_of_frames, self.channels, self.height, self.width), order="c"
        )
        # print(self.frames_buffer.flags)

        # Need to move RGB channel to last axis
        self.frames_buffer = np.moveaxis(self.frames_buffer, 1, 3)

        print(
            f"Loaded '{self.file_name}' - {self.number_of_frames} frames loaded into buffer"
        )

    def dump_frames(self, output_folder="video_frames") -> None:
        print(f"Dumping frames into '{output_folder}' folder")

        for frame_index, frame_buffer in enumerate(self.frames_buffer):
            im = Image.fromarray(frame_buffer)
            im.save(f"{output_folder}/{frame_index}.png")

            if frame_index % 100 == 0:
                print(f"--- At frame {frame_index}")

        print("Frame dump completed")

    def scan_for_shots(
        self,
        compare_method: int = cv.HISTCMP_BHATTACHARYYA,
        compare_threshold: float = 0.7,
    ) -> None:
        print(
            f"Detecting shots with '{FrameUtil.get_histogram_comparison_method_name(compare_method)}' with threshold {compare_threshold}"
        )

        # Clear shots array
        self.shots.clear()

        shot_start_frame_index = 0

        for frame_index in range(0, len(self.frames_buffer) - 1):
            current_frame: np.ndarray = self.frames_buffer[frame_index]
            next_frame: np.ndarray = self.frames_buffer[frame_index + 1]

            compare_val = FrameUtil.compare_histogram(
                current_frame, next_frame, compare_method
            )

            if compare_val > compare_threshold:
                self.shots.append((shot_start_frame_index, frame_index))
                shot_start_frame_index = frame_index + 1

                print(f"--- Frame {frame_index} -> {frame_index + 1} = {compare_val}")

        self.shots.append((shot_start_frame_index, len(self.frames_buffer) - 1))

        print(self.shots)

    def debug_override_shots(self, shots):
        self.shots = shots

    def scan_shots_for_logos(self, frame_skip=5):
        img = Image.fromarray(self.frames_buffer[2137])

        # img_base64 = base64.b64encode(data.tobytes())

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        image = vision.Image(content=buffer.getvalue())
        response = client.logo_detection(image=image)
        logos = response.logo_annotations
        print("Logos:")

        for logo in logos:
            print(logo.description)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

        img.close()

    def scan_shots_for_logos(self, frame_skip=100):

        detected_logos: Set[str] = set()

        buffer = BytesIO()
        for frame_index in range(0, self.number_of_frames, frame_skip):
            print(f"Checking frame {frame_index}")
            img = Image.fromarray(self.frames_buffer[frame_index])
            img.save(buffer, format="PNG")

            image = vision.Image(content=buffer.getvalue())
            response = client.logo_detection(image=image)
            logos = response.logo_annotations

            for logo in logos:
                detected_logos.add(logo.description)

            buffer.seek(0)
            buffer.truncate()

        print(detected_logos)

        # for shot_index, shot in enumerate(self.shots):
        #     shot_start_frame_index = shot[0]
        #     shot_end_frame_index = shot[1]
        #     for frame_index in range(
        #         shot_start_frame_index, shot_end_frame_index, frame_skip
        #     ):
        #         print(f"Checking frame {frame_index}")
        #         img = Image.fromarray(self.frames_buffer[frame_index])
        #         buffer = BytesIO()
        #         img.save(buffer, format="PNG")
        #         img.save(f"test{frame_index}.png", format="PNG")
        #         buffer.flush()
        #         image = vision.Image(content=buffer.getvalue())
        #         response = client.logo_detection(image=image)
        #         logos = response.logo_annotations

        #         if len(logos) > 0:
        #             if "subway" not in logos:
        #                 print(f"Shot {shot_index} has logos but Subway not in it")
        #                 print(logos)
        #                 break
        #             else:
        #                 print(f"Shot {shot_index} has Subway in it")
        #                 print(logos)
        #                 break

        # print(enco)
        # exit(0)


def main():
    pass
    rgb_video = RGBVideo("dataset-001/dataset/Videos/data_test1.rgb")
    # # rgb_video.dump_frames()
    # rgb_video.scan_for_shots()
    rgb_video.debug_override_shots(
        [
            (0, 1178),
            (1179, 2399),
            (2400, 2849),
            (2850, 4349),
            (4350, 5549),
            (5550, 5924),
            (5925, 5986),
            (5987, 5999),
            (6000, 8999),
        ]
    )
    rgb_video.scan_shots_for_logos()


if __name__ == "__main__":
    main()
