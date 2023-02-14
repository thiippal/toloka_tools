# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import json
import cv2
from pathlib import Path
from functions import json_to_cv2_bbox
from collections import Counter


"""
Usage:
    1. Add the colour scheme for each label into the file colours.json. Note that the RGB colour 
       values follow the order blue, green, red preferred by OpenCV.
    2. Run the script using the following command:
    
        python draw_outlines.py -i input.tsv -c "boxes_1" -g "input_image"
    
       For information on the arguments, run the command: python draw_outlines.py --help
    
    3. The script will write the annotated images on disk to the location determined by the 
       "output" argument.
"""

# Set up argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-t", "--input_tsv", required=True, help="Path to the TSV file containing the "
                                                         "output from Toloka.")
ap.add_argument("-c", "--columns", required=True, help="A string that names the columns that "
                                                       "contain the bounding boxes. If you have "
                                                       "annotations in multiple columns, separate "
                                                       "them with a space, e.g. 'box_1 box_2'.")
ap.add_argument("-g", "--group_by", required=True, help="A string that names the column that "
                                                        "contains values used for grouping the "
                                                        "annotations, e.g. individual images "
                                                        "that were annotated.")
ap.add_argument("-o", "--output", required=True, help="Path that determines where the output will "
                                                      "be saved.")

# Parse arguments
args = vars(ap.parse_args())

# Assign the output path to a variable and validate
output = Path(args['output'])

if not output.exists():

    exit("The output path does not exist. Exiting ...")

if not output.is_dir():

    exit("The output path is not a directory. Exiting ...")

# Split the columns that contain annotations into a list
args['columns'] = args['columns'].split()

# Read the input data and convert into a pandas DataFrame
df = pd.read_csv(args['input_tsv'], sep='\t')

# Group the data by the inputs
groups = df.groupby(args['group_by'])

# Initialise counter for different types of annotations drawn
counts = Counter()

# Loop over the groups
for img, annotations in groups:

    # Create a placeholder list for outlines
    outlines = []

    # Loop over each column that contains bounding boxes
    for annotation_col in args['columns']:

        # Fetch the annotations for each column from the group (annotations)
        for outline in annotations[annotation_col]:

            # Parse string into JSON
            outline = json.loads(outline)

            # Extend the list of outlines
            outlines.extend(outline)

    # Draw the outlines using OpenCV and return an annotated image
    ann_img, labels = json_to_cv2_bbox(img, outlines)

    # Update the counter
    counts.update(labels)

    # Construct output path
    output_path = output / Path(img.split('/')[-1].split('.')[0] + "_bboxes.png")

    # Write the annotated image to disk
    cv2.imwrite(str(output_path), ann_img)

print(f"Drew a total of {sum(counts.values())} bounding boxes: {counts.most_common()}")
