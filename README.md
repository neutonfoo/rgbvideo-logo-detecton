# How it works

1. `RGBVideo()`: Base video and ads-to-inject are all loaded as separate RGBVideos. The ads loaded as RGBVideos are labeled with ad_name, which is used later to match with detected logos. In each RGBVideo, the entire video and audio is loaded into 1 Shot instance (so len(self.shots) == 1). The audio buffer is also either padded or trimmed so that it matches the expected length based on the duration of the video.

2. `base_rgb_video.index_ads_to_inject(ads)`: The RGBVideo ads are indexed into the base RGBVideo for later use. Indexing simply puts the ads into a dictionary (in the base video) of ads where the company name is mapped to the ad RGBVideo.

3. `base_rgb_video.scan_video_for_shots()`: The single shot in the base video is broken up into individual shots. The Bhattacharyya distance between the histogram of all consecutive frame pairs is calculated. If the distance is above a specified threshold, those frames are considered as the end and start of two separate shots respectively. Add the completed shot to the base videos shots array (and copy the shots frame and audio buffers).

4. `base_rgb_video.merge_shots_using_audio_hueristic()`: Shorter, consecutive shots are merged if their audio amplitude is within a threshold. If an ad has many shots, this method aims to combine them into a single scene based on audio amplitude similarity. Longer shots are ignored.

5. `base_rgb_video.naive_ad_audio_classifier_and_merger()`: The average absolute amplitude of the longer shots is calculated. Shorter shots whose average absolute amplitude exceeds a specified threshold from the longer shot average are marked as ads. Consecutive shots marked as ads are then combined.

6. `base_rgb_video.verify_shots_continuity()`: Sanity check to ensure that book-keeping (namely, that the start frame and audio indices) were updated properly.

7. `base_rgb_video.scan_shots_for_logos()`: After the merging of shots from the previous step, all shots marked non-ads are scanned for logos through the Google Vision API. All logos (including those not needed / ones that we don't have ads for) are stored anyways. The order of logo detection is preserved.

8. `base_rgb_video.replace_ads_2()`: All the shots are looped and the shots marked as ads are removed and replaced by the ads-to-inject (based on the logos seen so far to that shot index). This behavior was clarified by a TA via email. For example: If shot 4 is the first marked ad, and logo 1 and logo 2 (in that order) were detected from shots 0 to 3, shot 4 is replaced by the ad for logo 1 and logo 2 respectively. The ads for logo 1 and logo 2 no longer need to be injected and are un-indexed.

9. `rebuild_frames_and_audio_buffers()`: The frame buffer on the original / parent RGBVideo is rebuilt from the self.shots array in preparation to be dumped.
