from __future__ import annotations

import os
from io import BytesIO
from typing import Dict, List, Set, Tuple

import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

load_dotenv()

client = vision.ImageAnnotatorClient.from_service_account_json(
    os.environ["GOOGLE_VISION_CREDS_JSON"]
)


class RGB_Util:
    @staticmethod
    def compare_histograms(
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

    @staticmethod
    def compare_audio_buffers(audio_buffer_1, audio_buffer_2):
        dist = np.linalg.norm(audio_buffer_1 - audio_buffer_2, ord=2)
        return dist


class Shot:
    def __init__(
        self,
        start_frame_index: int,
        frames_buffer: np.ndarray,
        start_audio_index: int,
        audio_buffer: np.ndarray,
    ):
        self.start_frame_index = start_frame_index
        self.frames_buffer = frames_buffer

        self.start_audio_index = start_audio_index
        self.audio_buffer = audio_buffer

        self.compute_average_absolute_amplitude()

    def compute_average_absolute_amplitude(self):
        self.average_absolute_amplitude = np.mean(np.abs(self.audio_buffer))

        # self.sign_changes = (
        #     np.sum(
        #         np.diff(np.sign(self.audio_buffer - np.mean(self.audio_buffer))) != 0
        #     )
        #     * 1
        # )

        # self.sign_changes = 0

        # is_sign_positive = self.audio_buffer[0] >= 0

        # for x in self.audio_buffer:
        #     if is_sign_positive and x < 0:
        #         self.sign_changes += 1
        #         is_sign_positive = False
        #     elif not is_sign_positive and x > 0:
        #         self.sign_changes += 1
        #         is_sign_positive = True

        # self.sign_changes /= len(self.audio_buffer)

    def normalize_audio_buffer_to_length(self, normalization_length: int):
        side_buffers = 0

        if len(self.audio_buffer) > normalization_length:
            side_buffers = int((len(self.audio_buffer) - normalization_length) / 2)

        self.normalized_audio_buffer = self.audio_buffer[
            side_buffers : side_buffers + normalization_length
        ]

        self.normalized_audio_buffer -= np.mean(self.normalized_audio_buffer)
        # print(self.normalized_audio_buffer)

        ## Normalize the values to be between [0,1]
        # This gets rid of negative values

        # Got overflow errors, need to convert to int32

        # # Normalize audio data
        # self.normalized_audio_buffer = (self.normalized_audio_buffer - min_amplitude) / (
        #     max_amplitude - min_amplitude
        # )

        # self.normalized_audio_buffer /=

        # for x in self.normalized_audio_buffer:
        #     print(x)

    def __repr__(self):
        return (
            f"V: {self.start_frame_index} -> {self.start_frame_index + len(self.frames_buffer) - 1}\n"
            f"A: {self.start_audio_index} -> {self.start_audio_index + len(self.audio_buffer) - 1})"
        )


class RGBVideo:
    def __init__(
        self,
        video_file_name: str,
        audio_file_name: str = "",
        width=480,
        height=270,
        channels=3,
        frame_rate=30,
        ad_name: str = "",
    ):
        self.video_file_name = video_file_name
        self.audio_file_name = audio_file_name
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_rate = frame_rate
        self.ad_name = ad_name

        self.shots: List[Shot] = []

        print(f"-- Processing '{self.video_file_name}' and '{self.audio_file_name}'")

        self.__read_video_file()
        self.__read_audio_file()

        # If the entire video needs to be analyzed for shots,
        # call scan_video_for_shots()
        # If not, the video will be treated as a single shot (like an ad),
        # which makes it easy to move into another video
        self.__init_shots()

        print("Done\n")

    def __read_video_file(self):
        self.frames_buffer = np.fromfile(self.video_file_name, dtype=np.uint8)

        # Calculate number of frames by dividing number of values by size of 1 frame
        self.number_of_frames = int(
            self.frames_buffer.shape[0] / (self.width * self.height * self.channels)
        )

        self.duration = self.number_of_frames / self.frame_rate

        # Shape = (9000, 3, 270, 480)
        self.frames_buffer = self.frames_buffer.reshape(
            (self.number_of_frames, self.channels, self.height, self.width)
        )

        # Need to move RGB channel to last axis
        # Shape = (9000, 270, 480, 3)
        self.frames_buffer = np.moveaxis(self.frames_buffer, 1, 3)

        print(
            f"'{self.video_file_name}' loaded into frames_buffer, {self.number_of_frames} frames loaded."
        )

    def __read_audio_file(self):
        if self.audio_file_name:
            self.audio_sample_rate, self.audio_buffer = wavfile.read(
                self.audio_file_name, True
            )
            expected_number_of_samples = int(self.audio_sample_rate * self.duration)

            print(
                f"'{self.audio_file_name}' loaded into audio_buffer, {len(self.audio_buffer)} samples loaded. (Expected {expected_number_of_samples} samples.)"
            )

            # We know the expected number of samples from the video data,
            # if the .wav file is missing samples, just pad on the right with 0s
            if self.audio_buffer.shape[0] < expected_number_of_samples:

                print(
                    f"- Missing {expected_number_of_samples - self.audio_buffer.shape[0]} samples. Applying zero-padding on right side."
                )

                self.audio_buffer = np.pad(
                    self.audio_buffer,
                    (0, expected_number_of_samples - self.audio_buffer.shape[0]),
                    "constant",
                    constant_values=(0, 0),
                )
        else:
            self.audio_sample_rate = 1
            expected_number_of_samples = int(self.audio_sample_rate * self.duration)
            self.audio_buffer = np.zeros(expected_number_of_samples)

        self.number_of_audio_samples = expected_number_of_samples
        # self.audio_buffer = self.audio_buffer.astype("float32")

    def __init_shots(self):
        # Initialize shots as the whole video
        self.shots.append(
            Shot(
                start_frame_index=0,
                frames_buffer=self.frames_buffer,
                start_audio_index=0,
                audio_buffer=self.audio_buffer,
            )
        )

    def index_ads_to_inject(self, ads_to_inject: List[RGBVideo] = []):
        # Cannot insert an ad into another ad. If ad_name is empty, this is the base video
        assert not self.ad_name

        self.ads_to_inject = {}

        for ad_to_inject in ads_to_inject:
            self.ads_to_inject[ad_to_inject.ad_name] = ad_to_inject

        print(f"-- Ads indexed to base video: {', '.join(self.ads_to_inject.keys())}\n")

    def dump_frames(self, output_folder="temp") -> None:
        print(f"-- Dumping frames into '{output_folder}' folder")

        for frame_index, frame_buffer in enumerate(self.frames_buffer):
            im = Image.fromarray(frame_buffer)
            im.save(f"{output_folder}/{frame_index}.png")

            if frame_index % 100 == 0:
                print(f"At frame {frame_index} of {self.frames_buffer.shape[0]}")

        print("Frame dump completed")

    def scan_video_for_shots(
        self,
        compare_method: int = cv.HISTCMP_BHATTACHARYYA,
        compare_threshold: float = 0.7,
    ) -> None:
        print(
            f"-- Detecting shots with '{RGB_Util.get_histogram_comparison_method_name(compare_method)}' with threshold {compare_threshold}"
        )

        # Only base video can be analyzed for shots
        assert not self.ad_name

        # Clear current shots array
        self.shots.clear()

        start_frame_index = 0
        for frame_index in range(0, len(self.frames_buffer) - 1):
            current_frame: np.ndarray = self.frames_buffer[frame_index]
            next_frame: np.ndarray = self.frames_buffer[frame_index + 1]

            compare_val = RGB_Util.compare_histograms(
                current_frame, next_frame, compare_method
            )

            if compare_val > compare_threshold:

                start_audio_index = self.__frame_to_audio_conversation(
                    start_frame_index
                )[0]
                end_audio_index = self.__frame_to_audio_conversation(frame_index)[1]
                self.shots.append(
                    Shot(
                        start_frame_index=start_frame_index,
                        frames_buffer=self.frames_buffer[
                            start_frame_index : frame_index + 1
                        ],
                        start_audio_index=start_audio_index,
                        audio_buffer=self.audio_buffer[
                            start_audio_index : end_audio_index + 1
                        ],
                    )
                )

                start_frame_index = frame_index + 1
                print(f"Frame {frame_index} -> {frame_index + 1} = {compare_val}")

        # Add remaining video as last shot
        start_audio_index = self.__frame_to_audio_conversation(start_frame_index)[0]
        end_audio_index = self.__frame_to_audio_conversation(frame_index)[1]

        self.shots.append(
            Shot(
                start_frame_index=start_frame_index,
                frames_buffer=self.frames_buffer[
                    start_frame_index : len(self.frames_buffer)
                ],
                start_audio_index=start_audio_index,
                audio_buffer=self.audio_buffer[start_audio_index : end_audio_index + 1],
            )
        )

        # Timestamp shots
        for shot in self.shots:
            shot_frame_start_index = shot.start_frame_index
            shot_frame_end_index = shot_frame_start_index + len(shot.frames_buffer) - 1

            print(
                f"Shot from frames {shot_frame_start_index} to {shot_frame_end_index}"
            )

        print()

        self.verify_shots_continuity()

    def verify_shots_continuity(self):
        print("-- Verifying shots continuity")

        prev_shot_end_frame_index = -1
        prev_shot_end_audio_index = -1

        has_shots_continuity_error = False

        for shot_index, shot in enumerate(self.shots):
            if shot.start_frame_index != prev_shot_end_frame_index + 1:
                has_shots_continuity_error = True
                print(
                    f"Video discontinuity from shot {shot_index - 1} to {shot_index} : {prev_shot_end_frame_index} -> {shot.start_frame_index}"
                )

            if shot.start_audio_index != prev_shot_end_audio_index + 1:
                has_shots_continuity_error = True
                print(
                    f"Audio discontinuity from shot {shot_index - 1} to {shot_index} : {prev_shot_end_audio_index} -> {shot.start_audio_index}"
                )

            prev_shot_end_frame_index = (
                shot.start_frame_index + len(shot.frames_buffer) - 1
            )
            prev_shot_end_audio_index = (
                shot.start_audio_index + len(shot.audio_buffer) - 1
            )

        if has_shots_continuity_error:
            print("-- Shots continuity error")
            exit(1)

        print("-- Shots continuity verified")
        print()

    def _debug_set_shots(self, shots: List[Tuple[int, int]]):

        print("-- Manually overriding shot detection")
        print("- For debug use only")

        self.shots.clear()
        for shot in shots:
            start_frame_index = shot[0]
            end_frame_index = shot[1]

            start_audio_index = self.__frame_to_audio_conversation(start_frame_index)[0]
            end_audio_index = self.__frame_to_audio_conversation(end_frame_index)[1]

            self.shots.append(
                Shot(
                    start_frame_index=start_frame_index,
                    frames_buffer=self.frames_buffer[
                        start_frame_index : end_frame_index + 1
                    ],
                    start_audio_index=start_audio_index,
                    audio_buffer=self.audio_buffer[
                        start_audio_index : end_audio_index + 1
                    ],
                )
            )

        # Timestamp shots
        for shot in self.shots:
            shot_frame_start_index = shot.start_frame_index
            shot_frame_end_index = shot_frame_start_index + len(shot.frames_buffer) - 1

            print(
                f"Shot from frames {shot_frame_start_index} to {shot_frame_end_index}"
            )

    def _debug_set_logo_detection(self, detected_logos_in_shots: List[List[str]] = []):
        print("-- Manually overriding logo detection")
        print("- For debug use only")

        self.detected_logos_in_shots = detected_logos_in_shots

        print("- Detection results")
        for shot_index, detected_logos_in_shot in enumerate(
            self.detected_logos_in_shots
        ):
            print(
                f"Detected logos in shot {shot_index}: {', '.join(detected_logos_in_shot)}"
            )

    def merge_shots_using_audio_hueristic(
        self,
        advertisement_length_tolerance: int = 20,
        advertisement_absolute_amplitude_delta_tolerance: int = 300,
    ):

        print("-- Merging shots with audio hueristics")

        advertisement_length_tolerance_in_frames = (
            advertisement_length_tolerance * self.frame_rate
        )

        print(
            f"- Advertisement length tolerance: {advertisement_length_tolerance}s = {advertisement_length_tolerance_in_frames} frames."
        )
        print(
            f"- Advertisement absolute amplitude delta tolerance: {advertisement_absolute_amplitude_delta_tolerance}."
        )

        # Loop backwards so consecutive shots from the back can be merged
        # Start from 1 before the end
        for shot_index in range(len(self.shots) - 2, -1, -1):
            shot = self.shots[shot_index]
            next_shot = self.shots[shot_index + 1]

            if len(shot.frames_buffer) > advertisement_length_tolerance_in_frames:
                print(f"Shot {shot_index} exceeds length tolerance. Skipping.")
                continue

            elif (
                len(shot.frames_buffer) + len(next_shot.frames_buffer)
                > advertisement_length_tolerance_in_frames
            ):
                print(
                    f"Merging of shots {shot_index} and {shot_index + 1} would exceed length tolerence. Skipping."
                )
                continue

            if (
                abs(
                    next_shot.average_absolute_amplitude
                    - shot.average_absolute_amplitude
                )
                <= advertisement_absolute_amplitude_delta_tolerance
            ):
                print(
                    f"Shots {shot_index} ({shot.average_absolute_amplitude}) and {shot_index + 1} ({next_shot.average_absolute_amplitude}) within audio tolerance. Merging."
                )
                self.merge_shots(shot_index, shot_index + 1)
            else:
                print(
                    f"Shots {shot_index} and shot {shot_index+1} exceed advertisement absolute amplitude delta tolerance. Skipping."
                )

        print()

    # Concatenates shot 2 to shot 1
    def merge_shots(self, shot_index_1: int, shot_index_2: int):
        # Concatenate base shot and concatenate shot
        base_shot = self.shots[shot_index_1]
        concatenate_shot = self.shots[shot_index_2]

        base_shot.frames_buffer = np.concatenate(
            [base_shot.frames_buffer, concatenate_shot.frames_buffer]
        )

        base_shot.audio_buffer = np.concatenate(
            [base_shot.audio_buffer, concatenate_shot.audio_buffer]
        )

        # if shot_index_1 < shot_index_2:
        #     # If shot 1 is before than shot 2
        #     # - Shots before shot 1 are unaffected
        #     # - Shots strictly between shot 1 and shot 2 have their start index pushed length of shot 2
        #     # - Shots after shot 2 are unaffected
        #     # - Shot 2 is removed
        #     for shot_index in range(shot_index_1 + 1, shot_index_2):

        #         print("Prob never called anyways 1")
        #         exit(1)

        #         self.shots[shot_index].start_audio_index += len(
        #             concatenate_shot.audio_buffer
        #         )
        #         self.shots[shot_index].start_frame_index += len(
        #             concatenate_shot.frames_buffer
        #         )

        # elif shot_index_1 > shot_index_2:
        #     # If shot 1 is after shot 2
        #     # Prob never gonna happen
        #     print("Prob never called anyways 2")
        #     exit(1)

        # Do not call self.remove_shot(). This should only be used if an entire shot is removed.
        # Since merging causes a previous shot to extend their time, removing the concantenated shot is enough
        # self.remove_shot() will result in all shots afer concatenate_shot_index to have their times adjusted

        del self.shots[shot_index_2]

    def remove_shot(self, remove_shot_index: int):
        print(f"Removing shot {remove_shot_index}")

        # If a shot is removed, get the frame and audio buffer lengths
        # and subtract the start frame index from all shots strictly after the shot_index by these lengths

        shot_frame_buffer_length = len(self.shots[remove_shot_index].frames_buffer)
        shot_audio_buffer_length = len(self.shots[remove_shot_index].audio_buffer)

        for shot_index in range(remove_shot_index + 1, len(self.shots)):
            self.shots[shot_index].start_frame_index -= shot_frame_buffer_length
            self.shots[shot_index].start_audio_index -= shot_audio_buffer_length

        del self.shots[remove_shot_index]

        print()

    def scan_shots_for_logos(
        self,
        frame_skip=100,
        label_font_size: int = 24,
    ):
        print(f"-- Scanning video for logos with frame skip {frame_skip}")

        # Since logos come in order, keep a list
        # Tuple[0] = Logo name, Tuple[1] = Earliest shot of detection
        self.detected_logos_in_shots: List[List[str]] = []
        detected_logos_set: Set[str] = set()

        # Create empty buffer to store images for base64 encoding
        buffer = BytesIO()
        label_font = ImageFont.truetype("Roboto-Regular.ttf", label_font_size)

        for shot_index, shot in enumerate(self.shots):
            print(f"- Scanning shot {shot_index}")

            # Detected logos in the current shot
            self.detected_logos_in_shots.append([])
            # = self.detected_logos_in_shots[shot_index]

            for frame_index in range(0, len(shot.frames_buffer), frame_skip):
                detected_logos_in_frame: List[str] = []

                img = Image.fromarray(shot.frames_buffer[frame_index])
                img.save(buffer, format="PNG")
                response = client.logo_detection(
                    image=vision.Image(content=buffer.getvalue())
                )
                logos = response.logo_annotations

                if len(logos) > 0:
                    img_logos_labeled = ImageDraw.Draw(img)

                    for logo in logos:

                        if logo.description not in detected_logos_set:
                            self.detected_logos_in_shots[shot_index].append(
                                logo.description
                            )
                            detected_logos_set.add(logo.description)

                        detected_logos_in_frame.append(logo.description)

                        print(
                            f"'{logo.description}' detected on frame {shot.start_frame_index + frame_index}"
                        )

                        logo_bounding_box = [
                            (
                                logo.bounding_poly.vertices[0].x,
                                logo.bounding_poly.vertices[0].y,
                            ),
                            (
                                logo.bounding_poly.vertices[2].x,
                                logo.bounding_poly.vertices[2].y,
                            ),
                        ]

                        img_logos_labeled.rectangle(
                            logo_bounding_box, outline="red", width=3
                        )

                    img_logos_labeled.text(
                        xy=(0, self.height - label_font_size),
                        text=", ".join(detected_logos_in_frame),
                        fill=(255, 0, 0),
                        font=label_font,
                    )
                    shot.frames_buffer[frame_index] = np.asarray(img)

                buffer.seek(0)
                buffer.truncate()

        print("- Detection results")

        for shot_index, detected_logos_in_shot in enumerate(
            self.detected_logos_in_shots
        ):
            print(
                f"Detected logos in shot {shot_index}: {', '.join(detected_logos_in_shot)}"
            )

    def naive_ad_audio_classifier_and_remover(
        self,
        advertisement_length_tolerance: int = 20,
        advertisement_absolute_amplitude_delta_tolerance: int = 300,
    ):
        print("-- Running naïve audio classifier")

        advertisement_length_tolerance_in_frames = (
            advertisement_length_tolerance * self.frame_rate
        )

        number_of_long_shots = 0
        average_long_shot_absolute_amplitude = 0

        for shot in self.shots:
            if len(shot.frames_buffer) > advertisement_length_tolerance_in_frames:
                number_of_long_shots += 1
                average_long_shot_absolute_amplitude += shot.average_absolute_amplitude

        average_long_shot_absolute_amplitude /= number_of_long_shots

        # Loop back and remove ads
        for shot_index in range(len(self.shots) - 1, -1, -1):
            shot = self.shots[shot_index]
            if len(shot.frames_buffer) <= advertisement_length_tolerance_in_frames:
                if (
                    abs(
                        shot.average_absolute_amplitude
                        - average_long_shot_absolute_amplitude
                    )
                    >= advertisement_absolute_amplitude_delta_tolerance
                ):
                    print(f"Shot {shot_index} classified as an ad")

                    # self.remove_shot(shot_index)

    def inject_ads(self):
        print("-- Injecting ads")

        # Have to keep track of number of inserted ads as shot_index offset
        number_of_inserted_ads = 0

        added_ad_names = set()

        for shot_index, detected_logos_in_shot in enumerate(
            self.detected_logos_in_shots
        ):
            for detected_logo_in_shot in detected_logos_in_shot:
                if detected_logo_in_shot in self.ads_to_inject.keys():
                    if detected_logo_in_shot in added_ad_names:
                        print(
                            f"{detected_logo_in_shot} advertisement already inserted. Skipping."
                        )
                        continue
                    print(
                        f"Inserting {detected_logo_in_shot} after shot {shot_index + number_of_inserted_ads}"
                    )

                    self.insert_rgb_video_after_shot_index(
                        self.ads_to_inject[detected_logo_in_shot],
                        shot_index + number_of_inserted_ads,
                    )
                    number_of_inserted_ads += 1

                    added_ad_names.add(detected_logo_in_shot)

        print()

    def insert_rgb_video_after_shot_index(
        self, rgb_video: RGBVideo, insert_shot_index: int
    ):

        # If inserting video, should only have 1 shot
        assert len(rgb_video.shots) == 1
        assert insert_shot_index >= 1

        shot_to_insert = rgb_video.shots[0]
        prev_shot = self.shots[insert_shot_index]

        # Reassign the start indices to 1 after the frame / audio
        shot_to_insert.start_frame_index = prev_shot.start_frame_index + len(
            prev_shot.frames_buffer
        )
        shot_to_insert.start_audio_index = prev_shot.start_audio_index + len(
            prev_shot.audio_buffer
        )

        for shot_index in range(insert_shot_index + 1, len(self.shots)):
            self.shots[shot_index].start_frame_index += len(
                shot_to_insert.frames_buffer
            )
            self.shots[shot_index].start_audio_index += len(shot_to_insert.audio_buffer)

        self.shots.insert(insert_shot_index + 1, shot_to_insert)

    # This is an expensive operation so should be done at the end
    def rebuild_frames_and_audio_buffers(self):
        self.__rebuild_frames_buffer()
        self.__rebuild_audio_buffer()
        self.verify_shots_continuity()

    def __rebuild_frames_buffer(self):
        print("-- Rebuilding video frames buffer")
        self.frames_buffer = np.concatenate([shot.frames_buffer for shot in self.shots])
        print("-- Frames buffer rebuilt")

    def __rebuild_audio_buffer(self):
        print("-- Rebuilding video audio buffer")
        self.audio_buffer = np.concatenate([shot.audio_buffer for shot in self.shots])
        print("-- Audio buffer rebuilt")

    def dump_rgb_file(self, dump_rgb_file_name: str):

        assert dump_rgb_file_name.endswith(".rgb")

        print(f"-- Dumping {dump_rgb_file_name} file")

        # Create a temporary buffer with the RGB axis moved back and dump
        dump_frames_buffer = np.moveaxis(self.frames_buffer, 3, 1)
        dump_frames_buffer.tofile("new.rgb")

        print(f"-- {dump_rgb_file_name} file dumped")

    def dump_audio_file(self, dump_audio_file_name: str):
        assert dump_audio_file_name.endswith(".wav")

        print(f"-- Dumping {dump_audio_file_name} file")

        wavfile.write("new.wav", self.audio_sample_rate, self.audio_buffer)

        print(f"-- {dump_audio_file_name} file dumped")

    def __frame_to_time_conversion(self, frame_index, round_time: bool = False):
        t = frame_index / self.frame_rate
        if round_time:
            return round(t, 2)
        return t

    def __frame_to_audio_conversation(self, frame_index) -> Tuple[int, int]:
        # Returns [start_index, end_index]
        # Example: If FPS = 30 and SampleRate = 48000
        # 1 Frame = 48000 / 30 = 1600 sample

        t = self.__frame_to_time_conversion(frame_index)

        start_t = round(t * self.audio_sample_rate)
        end_t = start_t + round(self.audio_sample_rate / self.frame_rate) - 1
        # Subtract 1 because get to the sample before the first sample of the next frame

        return (start_t, end_t)


def folder_structure_message():
    print("-- Add the following .env variables")
    print('GOOGLE_VISION_CREDS_JSON="creds/google_vision_creds.json"')
    print()
    print("-- Please create the following folder structure")
    print("./creds/google_vision_creds.json")
    print("./dataset-001/dataset1/{Ads,Brand Images,Videos}")
    print("./dataset-002/dataset2/{Ads,Brand Images,Videos}")
    print("./dataset-003/dataset3/{Ads,Brand Images,Videos}")
    print("./video_frames/temp")
    print("./video_frames/video_frames_1")
    print("./video_frames/video_frames_2")
    print("./video_frames/video_frames_3")


def run_dataset(
    base_rgb_video: RGBVideo,
    ads: List[RGBVideo],
    # Refer to https://docs.opencv.org/4.5.5/d8/dc8/tutorial_histogram_comparison.html for compare methods.
    # IMO this is the best
    scan_video_for_shots__compare_method: int = cv.HISTCMP_BHATTACHARYYA,
    scan_video_for_shots__compare_threshold: float = 0.7,
    merge_shots_using_audio_hueristic__advertisement_length_tolerance: int = 20,
    merge_shots_using_audio_hueristic__advertisement_absolute_amplitude_delta_tolerance: int = 300,
    naive_ad_audio_classifier_and_remover__advertisement_length_tolerance: int = 20,
    naive_ad_audio_classifier_and_remover__advertisement_absolute_amplitude_delta_tolerance: int = 300,
    scan_shots_for_logos__frame_skip: int = 100,
    scan_shots_for_logos__label_font_size: int = 20,
    dump_rgb_file__dump_rgb_file_name="new.rgb",
    dump_audio_file__dump_audio_file_name="new.wav",
):
    # Index ads to inject
    base_rgb_video.index_ads_to_inject(ads)

    # Scan for shots
    base_rgb_video.scan_video_for_shots(
        compare_method=scan_video_for_shots__compare_method,
        compare_threshold=scan_video_for_shots__compare_threshold,
    )

    # Merge shorter, audio-similar shots
    base_rgb_video.merge_shots_using_audio_hueristic(
        advertisement_length_tolerance=merge_shots_using_audio_hueristic__advertisement_length_tolerance,
        advertisement_absolute_amplitude_delta_tolerance=merge_shots_using_audio_hueristic__advertisement_absolute_amplitude_delta_tolerance,
    )

    # Remove shorter shots that are audio-different from the longer shots
    base_rgb_video.naive_ad_audio_classifier_and_remover(
        advertisement_length_tolerance=naive_ad_audio_classifier_and_remover__advertisement_length_tolerance,
        advertisement_absolute_amplitude_delta_tolerance=naive_ad_audio_classifier_and_remover__advertisement_absolute_amplitude_delta_tolerance,
    )

    # Sanity check
    base_rgb_video.verify_shots_continuity()

    # Scan shots for logos and index
    base_rgb_video.scan_shots_for_logos(
        frame_skip=scan_shots_for_logos__frame_skip,
        label_font_size=scan_shots_for_logos__label_font_size,
    )

    # Inject ads
    base_rgb_video.inject_ads()

    # Rebuild the RGBVideo frame and audio buffers.
    # - This is an expensive operation so only call once at the bottom
    # - To verify that the steps succeeded, it's faster to run the santiy check: verify_shots_continuity()
    base_rgb_video.rebuild_frames_and_audio_buffers()

    # Dump the files
    base_rgb_video.dump_rgb_file(dump_rgb_file_name=dump_rgb_file__dump_rgb_file_name)
    base_rgb_video.dump_audio_file(
        dump_audio_file_name=dump_audio_file__dump_audio_file_name
    )


# def test_dataset_001():
#     base_rgb_video = RGBVideo(
#         video_file_name="dataset-001/dataset1/Videos/data_test1.rgb",
#         audio_file_name="dataset-001/dataset1/Videos/data_test1.wav",
#     )
#     starbucks_ad = RGBVideo(
#         video_file_name="dataset-001/dataset1/Ads/Starbucks_Ad_15s.rgb",
#         audio_file_name="dataset-001/dataset1/Ads/Starbucks_Ad_15s.wav",
#         ad_name="Starbucks",
#     )
#     subway_ad = RGBVideo(
#         video_file_name="dataset-001/dataset1/Ads/Subway_Ad_15s.rgb",
#         audio_file_name="dataset-001/dataset1/Ads/Subway_Ad_15s.wav",
#         ad_name="Subway",
#     )

#     base_rgb_video.index_ads_to_inject([starbucks_ad, subway_ad])
#     # base_rgb_video.scan_video_for_shots()
#     base_rgb_video._debug_set_shots(
#         [
#             (0, 1178),
#             (1179, 2399),
#             (2400, 2849),
#             (2850, 4349),
#             (4350, 5549),
#             (5550, 5924),
#             (5925, 5986),
#             (5987, 5999),
#             (6000, 8999),
#         ]
#     )

#     base_rgb_video.merge_shots_using_audio_hueristic()
#     base_rgb_video.naive_ad_audio_classifier_and_remover()
#     base_rgb_video.verify_shots_continuity()
#     # base_rgb_video.scan_shots_for_logos()
#     # base_rgb_video._debug_set_logo_detection(
#     #     [[], ["Subway"], [], ["Starbucks"], [], []]
#     # )

#     base_rgb_video.inject_ads()
#     base_rgb_video.rebuild_frames_and_audio_buffers()
#     # base_rgb_video.dump_frames()
#     base_rgb_video.dump_rgb_file()
#     base_rgb_video.dump_audio_file()


def main():
    # folder_structure_message()
    # exit(0)

    base_rgb_video = RGBVideo(
        video_file_name="dataset-001/dataset1/Videos/data_test1.rgb",
        audio_file_name="dataset-001/dataset1/Videos/data_test1.wav",
    )
    starbucks_ad = RGBVideo(
        video_file_name="dataset-001/dataset1/Ads/Starbucks_Ad_15s.rgb",
        audio_file_name="dataset-001/dataset1/Ads/Starbucks_Ad_15s.wav",
        ad_name="Starbucks",
    )
    subway_ad = RGBVideo(
        video_file_name="dataset-001/dataset1/Ads/Subway_Ad_15s.rgb",
        audio_file_name="dataset-001/dataset1/Ads/Subway_Ad_15s.wav",
        ad_name="Subway",
    )

    run_dataset(base_rgb_video=base_rgb_video, ads=[starbucks_ad, subway_ad])
    # test_dataset_002()
    # test_dataset_003()


if __name__ == "__main__":
    # Need to proccess CLI
    main()
