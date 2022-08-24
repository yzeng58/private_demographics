import os, torchvision, random, argparse
import pandas as pd
import torchvision.transforms as transforms
from settings import *

def get_outlier_transforms(blur=True, colorjitter=True, posterize=True, flip_pct=2):
        outlier_transforms = []
        if posterize: outlier_transforms.append(transforms.RandomPosterize(bits=4, p=1))
        if colorjitter: outlier_transforms.append(transforms.ColorJitter())
        if blur: outlier_transforms.append(transforms.GaussianBlur(kernel_size=11))
        outlier_transforms.extend(['flip_label']*flip_pct)
        return outlier_transforms

def generate_outliers(
    dataset_name = 'waterbirds',
): # only works for waterbirds for now
    root_dir = '/dccstor/storage/balanceGroups/data'
    if dataset_name == 'waterbirds':
        data_dir = '/dccstor/storage/balanceGroups/data/waterbirds/waterbird_complete95_forest2water2'
    out_dir = os.path.join(root_dir, dataset_name, 'outliers')        
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    transf = get_outlier_transforms()
    transf.extend([None for i in range(100 - len(transf))])

    metadata_df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    new_labels = []
    y_c = metadata_df['y'].values
    g_c = metadata_df['place'].values
    names_c = metadata_df['img_filename'].values
    splits = metadata_df['split']

    for i in range(len(metadata_df.index)):
        if y_c[i] == 0:
            if g_c[i] == 0:
                new_labels.append(0)
            else:
                new_labels.append(1)
        else:
            if g_c[i] == 0:
                new_labels.append(2)
            else:
                new_labels.append(3)

    for j, image_name in enumerate(names_c):
        subdir = image_name.split("/")[0]
        sub_path = os.path.join(out_dir, subdir)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

        input_path = os.path.join(data_dir, image_name)
        img_file = torchvision.io.read_image(input_path)
        new_img = None

        # if not part of test set
        if splits[j] != 2:
            t_n = random.randint(0, len(transf) - 1)
            t = transf[t_n]

            if t:
                print("Applying transform", t, "to file", image_name)
                if t == 'flip_label':
                    y_c[j] = 1 - y_c[j]
                else:
                    new_img = t(img_file)
                new_labels[j] = 4
        if new_img == None: new_img = img_file
        outpath = os.path.join(out_dir, image_name)
        torchvision.io.write_jpeg(new_img, outpath)

    metadata_df["group_labels"] = new_labels
    metadata_df["y"] = y_c
    metadata_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="waterbirds", choices = datasets)
    args = parser.parse_args()

    generate_outliers(
        args.dataset,
    )