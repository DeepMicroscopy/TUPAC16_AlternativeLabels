# Alternative training set for the TUPAC16 auxiliary mitotic figure data set

This repository contains our alternative training set labels for the auxiliary mitotic figure dataset of the [TUPAC 2016 challenge](http://tupac.tue-image.nl/). 

For details about the creation of this data set and the evaluation carried out, please have a look at our paper:
- Christof A. Bertram, Mitko Veta, Christian Marzahl, Nikolas Stathonikos, Andreas Maier,Robert Klopfleisch, and Marc Aubreville: Are pathologist-defined labels reproducible? Comparison of the TUPAC16 mitotic figure dataset with an alternative set of labels


In this repository, you can find the following content:

## Stitching 

In order to facilitate labeling and later training, we stitched all patches given by the TUPAC challenge which correspond to one case into a single big TIFF image (in pyramidal format). 
This stitching can be reperformed by you, using the following notebook:
- [StitchTUPAC16.ipynb](StitchTUPAC16.ipynb)

Unstitching of the resulting database was done using the following notebook:
- [TransformDatabaseToCsvForPatches.ipynb](TransformDatabaseToCsvForPatches.ipynb)

## The databases

### Original CSV format

We provide the databases in the original format given by the TUPAC16 challenge in the subfolder [TUPAC_AL/](TUPAC_AL/). There you can find two directories:
- [TUPAC_AL/mitoses_ground_truth](TUPAC_AL/mitoses_ground_truth): A folder containing annotations for each original tile, subgrouped into the directories as the original images.
- [TUPAC_AL/nonmitoses_ground_truth](TUPAC_AL/nonmitoses_ground_truth): A folder in the same format, but not referencing mitotic figures but hard examples that can be used for facilitated training of a pipeline.

In each of the folders, you can find CSV files with "," as delimiter, representing the y-coordinate and the x-coordinate of annotated cells. Each new line represents a new annotation.
Please note that the database in CSV format corresponds to the original images, not the stitched versions.

### SQLITE database format

We trained our pipelines using fast.ai and a SQLITE3 database to facilitate the access. If you want to work with these pipelines, please download the [https://github.com/DeepPathology/SlideRunner]SlideRunner libraries, as they will be required for accessing the databases.

The annotations (together with the complete annotation record from each pathologist) are stored in this database. There are two versions of the database:
- [TUPAC_alternativeLabels_training.sqlite](TUPAC_alternativeLabels_training.sqlite) The database that was created purely manually (only using SlideRunner as annotation tool for blinding and guiding the experts during screening).
- [TUPAC_alternativeLabels_augmented_training.sqlite](TUPAC_alternativeLabels_augmented_training.sqlite) The database that contains additional mitotic figures and nonmitotic cells, which is a result of using RetinaNet and a subsequent ResNet18-classifier to find previously unannotated cells. Please note that all cells had to be reconfirmed by blind assessment by two experts in order to be included.
- [TUPAC_stitched.sqlite](TUPAC_stitched.sqlite) Original labels provided by the TUPAC16 challenge, aligned to the stitched layout on tiled WSI.

Note that the annotations all correspond to the stitched images created above.

Alternatively to using the SlideRunner libraries, you can also join the tables <i>Annotations</i> and <i>Annotations_coordinates</i> based on the field Annotations.uuid == Annotations_coordinates.annoid and use this information filtered by the field Annotations.slide. 

## Training of RetinaNet as baseline

We provide both notebooks that we used to train our baseline results on the alternative label set.

- [RetinaNet-TUPACoriginal-OrigSplit.ipynb](RetinaNet-TUPACoriginal-OrigSplit.ipynb) RetinaNet, patch size 512x512px, trained on original labels of TUPAC16
- [RetinaNet-TUPAC-AL-OrigSplit.ipynb](RetinaNet-TUPAC-AL-OrigSplit.ipynb) RetinaNet, patch size 512x512px, trained on our alternative labels of TUPAC16

## Crossvalidation

We share the notebooks used to train RetinaNet by Lin et al. on the original TUPAC16 labels and the new, alternative labels. This was performed in a cross-validation (based on the split given in TUPAC_Trainingset_Folds.p) with three folds.

### Training on original TUPAC:
- [RetinaNet-TUPAC-CrossValidationTrainingset_Batch1.ipynb](RetinaNet-TUPAC-CrossValidationTrainingset_Batch1.ipynb)
- [RetinaNet-TUPAC-CrossValidationTrainingset_Batch2.ipynb](RetinaNet-TUPAC-CrossValidationTrainingset_Batch2.ipynb)
- [RetinaNet-TUPAC-CrossValidationTrainingset_Batch3.ipynb](RetinaNet-TUPAC-CrossValidationTrainingset_Batch3.ipynb)

### Training on TUPAC alternative labels:
- [RetinaNet-TUPAC_AL-CrossValidationTrainingset_Batch1.ipynb](RetinaNet-TUPAC_AL-CrossValidationTrainingset_Batch1.ipynb)
- [RetinaNet-TUPAC_AL-CrossValidationTrainingset_Batch2.ipynb](RetinaNet-TUPAC_AL-CrossValidationTrainingset_Batch2.ipynb)
- [RetinaNet-TUPAC_AL-CrossValidationTrainingset_Batch3.ipynb](RetinaNet-TUPAC_AL-CrossValidationTrainingset_Batch3.ipynb)

