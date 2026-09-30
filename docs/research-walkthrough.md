# From a salmon image to a measurable shape

The interesting part of this project is the step after detection: using the locations of 20 anatomical points to study shape. A bounding box locates the fish. Keypoints provide measurements within that box.

## Read the implementation in this order

| Step | Implementation | What to look for |
| --- | --- | --- |
| Check the input | [validate_dataset.py](../scripts/validate_dataset.py) | Pair images by filename, require 65 fields per object, reject nonfinite values and invalid visibility |
| Configure the task | [config.yaml](../config.yaml) | One fish class and `kpt_shape: [20, 3]` |
| Start training | [train.py](../scripts/train.py) | Resolve local paths, validate both splits, seed 42, disable flips |
| Load predictions | [predict.py](../scripts/predict.py) | Require local weights and check the model has 20 keypoints |
| Examine geometry | [distance_pairs.py](../Geomtircal%20analysis/distance_pairs.py) | Euclidean distances between selected point pairs and baseline summary statistics |
| Inspect experiments | [slope_triangles_ratio_file.py](../Geomtircal%20analysis/slope_triangles_ratio_file.py) | Manually selected examples, slopes and ratios |

The helpers provide a reproducible entry point. The geometry files preserve the research exploration; they are not a fully connected inference pipeline.

## What an annotation means

A YOLO pose row has a class ID, four bounding-box values, and twenty coordinate/visibility triplets:

`class cx cy width height x0 y0 v0 ... x19 y19 v19`

That is 65 numbers. Bounding-box and point coordinates are normalised to the image. Visibility values are 0, 1 or 2; `validate_row` checks their range, not whether a landmark is anatomically correct.

For geometry in image space, convert x and y using the actual image width and height before comparing distances or angles. Unequal scaling along the two axes changes slopes and angles. A ratio can remove a shared scale factor, but it does not automatically remove camera perspective or pose.

The numbered image in the README shows dataset annotations. It is useful for inspecting point placement, and it is deliberately not labelled as a model prediction.

## Why these implementation choices matter

| Choice | Reason | Boundary |
| --- | --- | --- |
| Validate before training | Catch missing labels and malformed rows before an expensive run | A valid row can still contain a wrong annotation |
| Disable horizontal and vertical flips | Landmark order must remain anatomically meaningful | Re-enable only after checking the point permutation |
| Keep train and validation filenames separate | Prevent exact filename overlap | Related frames or the same fish may still appear in both splits |
| Check the trained model's keypoint shape | Prevent an unrelated pose checkpoint from being treated as this model | Matching shape does not prove good localisation |
| Retain baseline CSV files | Make the geometric exploration inspectable | Baselines and thresholds need an independent evaluation |

## What would make an evaluation convincing

Freeze the environment, seed, weights and split. Separate recordings or individual fish across the split where identifiers are available. Report localisation error for visible points, then evaluate the geometric decision rule separately: false positives, missed examples and failure cases.

There are no bundled trained weights or verified held-out deformity metrics. The offline tests establish input-format behaviour. They do not establish model accuracy or a reliable deformity diagnosis.
