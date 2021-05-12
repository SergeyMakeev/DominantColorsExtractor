import cv2
import os
import numpy as np
from sklearn.cluster import KMeans


def get_filename_without_ext(path):
    path = normalize_file_path(path)
    prefix_index = path.rfind('/')
    if prefix_index >= 0:
        path = path[prefix_index+1:]

    postfix_index = path.rfind('.')
    if postfix_index >= 0:
        path = path[:postfix_index]

    return path


def normalize_file_path(path):
    path = path.replace('\\', '/')

    # remove double slashes
    while path.find('//') >= 0:
        path = path.replace('//', '/')

    return path


def ensure_path_exist(file_path: str) -> str:
    dir_name = os.path.dirname(file_path)
    if not dir_name:
        return ""

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    return dir_name


def process_file(file_name: str, palette_file_name: str, resize: bool):
    print("File: " + file_name)
    num_colors = 16
    image = cv2.imread(file_name)

    if resize:
        print("Resize")
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    print("Reshape")
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))

    print("Fit")
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    print("Save palette")
    colors = kmeans.cluster_centers_.astype(int)

    image_palette = np.zeros((32, 32 * num_colors, 3), np.uint8)

    ox = 0
    for color in colors:
        print(color)
        for x in range(32):
            for y in range(32):
                image_palette[y, ox + x] = color
        ox = ox + 32

    ensure_path_exist(palette_file_name)
    cv2.imwrite(palette_file_name, image_palette)
    # cv2.imshow("Blank Image", image_palette)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def process_all_images():
    palettes_directory = "./palettes"
    directory = "./images"

    for file in os.listdir(palettes_directory):
        file_name = os.path.join(palettes_directory, file)
        print("Remove: " + file_name)
        os.remove(file_name)

    for root, dirs, files in os.walk(directory, topdown=True):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.png'):
                src_file_name = directory + "/" + name
                dst_file_name = palettes_directory + "/" + get_filename_without_ext(src_file_name) + "_palette.png"
                process_file(src_file_name, dst_file_name, True)


def concat_all_palettes():
    directory = "./palettes"
    images = []
    for root, dirs, files in os.walk(directory, topdown=True):
        for name in files:
            if name.endswith('_palette.png'):
                src_file_name = directory + "/" + name
                image = cv2.imread(src_file_name)
                images.append(image)

    im_v = cv2.vconcat(images)
    cv2.imwrite("combined_palette.png", im_v)


def main():
    process_all_images()
    concat_all_palettes()
    process_file("combined_palette.png", "final_palette.png", False)


main()
