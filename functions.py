# -*- coding: utf-8 -*-

import numpy as np
import cv2
import json
from skimage import io


# Load colour scheme from the JSON file
with open('colours.json') as json_f:

    # Assign colours to a dictionary
    COLOURS = json.loads(json_f.read())

    # Convert lists into tuples
    COLOURS = {k: tuple(v) for k, v in COLOURS.items()}


def convert_box(outlines, h, w):

    x1, y1 = int(round(w * outlines['left'])), int(round((h * outlines['top']), 0))
    elem_w, elem_h = int(round((w * outlines['width']), 0)), int(round((h * outlines['height']), 0))

    x2, y2 = elem_w + x1, elem_h + y1

    return x1, y1, x2, y2


def convert_poly(outlines, h, w):

    points = [[int(round((p['left']) * w, 0)), int(round((p['top'] * h), 0))]
              for p in outlines['points']]

    points = np.array(points, np.int32)

    return points


def json_to_cv2_bbox(image, outlines):

    # Read URL using skimage
    img = io.imread(image)
    h, w = img.shape[0], img.shape[1]

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back for coloured annotations
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Set up counter
    labels = []

    # Draw outlines
    for o in outlines:

        if o['shape'] == 'rectangle':

            x1, y1, x2, y2 = convert_box(o, h, w)

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          COLOURS[o['label']] if 'label' in o else (0, 255, 0), 2, cv2.LINE_AA)

        if o['shape'] == 'polygon':

            points = convert_poly(o, h, w)

            cv2.polylines(img, [points], True,
                          COLOURS[o['label']] if 'label' in o else (0, 255, 0), 2, cv2.LINE_AA)

        labels.append(o['label'] if 'label' in o else 'no_label')

    return img, labels
