# How it works

1. `RGBVideo()` Base video and ads to insert are all loaded as RGBVideos. Ads are labeled with ad_name, which is used to identify from logo detection. The entire video and audio is is initially loaded as a single shot (so len(self.shots) == 1).

2. `base_rgb_video.index_ads_to_inject(ads)`: Ads are indexed into the base video for later use.

3. `base_rgb_video.scan_video_for_shots()`: Bhattacharyya distance between the histogram of all consecutive frame pairs is calculated. If the distance is above a threshold, consider them two different shots.

4. `base_rgb_video.merge_shots_using_audio_hueristic()`: Shorter, consecutive shots are merged if their audio amplitude is within a threshold. If an ad has many shots, this method aims to combine them into a single shot based on thier audio similarity.

5. `base_rgb_video.naive_ad_audio_classifier_and_merger()`: The average absolute amplitude of longer shots is calculated. Shorter shots whose average absolute amplitude is too different from the longer shot average are marked as ads. Consecutive shots marked as ads are combined.

6. `base_rgb_video.verify_shots_continuity()`: Sanity check to ensure that book-keeping (namely, that the start frame and audio indices) were updated properly.

7. `base_rgb_video.scan_shots_for_logos()`: After the merging of shots from the previous step, all shots marked non-ads are scanned for logos through the Google Vision API. All logos (including those not needed / ones that we don't have ads for) are stored anyways. Order of ad detection preserved.

8. `base_rgb_video.replace_ads()`: The logos are looped through one-by-one and shots marked as ads are replaced in the base video by the corresponding ads.

9. `rebuild_frames_and_audio_buffers()`: The frame buffer on the original / parent RGBVideo is rebuilt from the self.shots array in preparation to be dumped.
