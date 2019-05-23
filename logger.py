"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import tensorflow as tf
import numpy as np
import scipy.misc
import os
import shutil

from io import BytesIO  # Python 3.x
from subprocess import call


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        self.writer = tf.summary.FileWriter(log_dir, filename_suffix=suffix)
        self.log_dir = log_dir

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def video_summary(self, tag, videos, step):

        sh = list(videos.shape)
        sh[-1] = 1

        separator = np.zeros(sh, dtype=videos.dtype)
        videos = np.concatenate([videos, separator], axis=-1)

        img_summaries = []

        temp_dir = os.path.join(self.log_dir, 'temp')
        vid_dir = os.path.join(self.log_dir, 'vid_{}'.format(step))
        os.makedirs(vid_dir, exist_ok=True)

        for i, vid in enumerate(videos):
            # Concat a video
            v = vid.transpose(1, 2, 3, 0)
            v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]

            os.makedirs(temp_dir, exist_ok=True)
            # write it to video file
            for frame_idx in range(len(v)):
                file_name = os.path.join(temp_dir,
                                         "{:06d}.png".format(frame_idx))
                img = v[frame_idx]
                scipy.misc.imsave(file_name, img, format="png")
            call(["avconv", "-y", "-f", "image2", "-i",
                  os.path.join(temp_dir, "%06d.png"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "15",
                  "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                  "-loglevel", "panic",
                  os.path.join(vid_dir, "vid_{}.mp4".format(i))
                ])
            shutil.rmtree(temp_dir)

            s = BytesIO()
            img = np.concatenate(v, axis=1)[:, :-1, :]

            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
