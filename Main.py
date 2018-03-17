import cv2
import time

from Classifier import Classifier
from Dataset import Dataset
from FeatureCombiner import FeatureCombiner
from FramesetProcessor import FramesetProcessor
from HOGExtractor import HOGExtractor
from HistExtractor import HistExtractor
from SpacialExtractor import SpacialExtractor
import matplotlib.pyplot as plt


def process_frames(input_folder, output_folder, classifier, l=None, f=None):
    images_path = FramesetProcessor.load_frames(input_folder)
    processor = FramesetProcessor(classifier)
    if f is None:
        f = 0
    frame_no = f
    for path in images_path[f:l]:
        img = cv2.imread(path)
        img = processor.process_frame_bgr(img)
        cv2.imwrite(output_folder + "/" + str(frame_no) + ".jpg", img)
        print(frame_no)
        frame_no = frame_no + 1


def process_video(input_vid, output_vid, classifier):
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip
    v_out = output_vid
    clip1 = VideoFileClip(input_vid)
    processor = FramesetProcessor(classifier)

    def process(frame):
        time.sleep(0.02)
        return processor.process_frame_rgb(frame)

    white_clip = clip1.fl_image(process)  # NOTE: this function expects color images!!
    white_clip.write_videofile(v_out, audio=False)


if __name__ == '__main__':
    # dataset = Dataset.from_path(max_size=4000, pickle_path="./dataset/dataset.p")
    # combiner = FeatureCombiner([SpacialExtractor(), HistExtractor(), HOGExtractor()],
    #                            pickle_path="./dataset/combiner.p")
    # classifier = Classifier(combiner)
    # classifier.train(dataset, pickle_path="./dataset/classifier.p")

    classifier = Classifier.from_pickle("./dataset/classifier.p")
    # process_frames("./video_frames/project_video/" ,"./video_frames/project_video_annotated/",classifier, f=950)
    process_video("./project_video.mp4", './project_video_annotated.mp4', classifier)
