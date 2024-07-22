import fiftyone as fo
import pandas as pd
from fiftyone import ViewField as F

my_classes = ["Hammer", "Axe", "Book"]
export_dir = "../../dataset/"

dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    classes=my_classes,
    max_samples=1000,
    shuffle=True
)

dataset = dataset.filter_labels("ground_truth", F("label").is_in(my_classes))

patches = dataset.to_patches("ground_truth")

patches.export(
    export_dir=export_dir,
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    label_field="ground_truth",
)