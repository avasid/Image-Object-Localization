import pickle

import numpy as np
from PIL import Image


def normalize(image):
    mean_lst = []
    std_list = []
    for channel in range(3):
        mean = np.mean(image[:, :, channel])
        std = np.std(image[:, :, channel])
        image[:, :, channel] = (image[:, :, channel] - mean) / std
        mean_lst.append(mean)
        std_list.append(std)
    return image, mean_lst, std_list


def convert(RESOLUTION):
    id_to_data = {}
    id_to_size = {}
    id_to_mean = {}
    id_to_std = {}

    with open("./src/data/images.txt") as f:
        lines = f.read().splitlines()
        len_lines = len(lines)
        for line in lines:
            id, path = line.split(" ", 1)
            image = Image.open("./src/data/images/" + path).convert('RGB')
            id_to_size[int(id)] = np.array(image, dtype=np.float32).shape[0:2]
            image = image.resize((RESOLUTION, RESOLUTION))
            image = np.array(image, dtype=np.float32)
            image = image / 255
            image, mean_list, std_list = normalize(image)
            id_to_data[int(id)] = image
            id_to_mean[int(id)] = mean_list
            id_to_std[int(id)] = std_list
            print("Processing images...  " + str(id) + "/" + str(len_lines))

    id_to_data = np.array(list(id_to_data.values()))
    id_to_mean = np.array(list(id_to_mean.values()))
    id_to_std = np.array(list(id_to_std.values()))
    id_to_size = np.array(list(id_to_size.values()))
    f = open("./id_to_data", "wb+")
    pickle.dump(id_to_data, f, protocol=4)
    f = open("./id_to_size", "wb+")
    pickle.dump(id_to_size, f, protocol=4)
    f = open("./id_to_mean", "wb+")
    pickle.dump(id_to_mean, f, protocol=4)
    f = open("./id_to_std", "wb+")
    pickle.dump(id_to_std, f, protocol=4)

    id_to_box = {}

    with open("./src/data/bounding_boxes.txt") as f:
        lines = f.read().splitlines()
        for line in lines:
            id, box = line.split(" ", 1)
            box = np.array([float(i) for i in box.split(" ")], dtype=np.float32)
            box[0] = box[0] / id_to_size[int(id) - 1][1] * RESOLUTION
            box[1] = box[1] / id_to_size[int(id) - 1][0] * RESOLUTION
            box[2] = box[2] / id_to_size[int(id) - 1][1] * RESOLUTION
            box[3] = box[3] / id_to_size[int(id) - 1][0] * RESOLUTION
            id_to_box[int(id)] = box

    id_to_box = np.array(list(id_to_box.values()))
    f = open("./id_to_box", "wb+")
    pickle.dump(id_to_box, f, protocol=4)
