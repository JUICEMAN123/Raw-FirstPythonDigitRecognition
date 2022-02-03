from PIL import Image
import os
import math

DATA = 'data/'
FEATURES = 'features/'
OUTPUTS = 'outputs/'
MNIST = 'mnist/'

def read_image(path):
    return Image.open(path)

def read_image_data(path):
    return read_image(path).getdata();

def read_image_data_in_grayscale(path): #matrix form
    img_gs = read_image(path).convert('LA')
    img_gs_data = [pixel[0] for pixel in img_gs.getdata()]
    w, h = img_gs.size
    return unflatten(img_gs_data, w)

def read_image_data_in_grayscale_normalized(path):
    return normalize_grayscale_image_data(read_image_data_in_grayscale(path))

def read_mnist_images(path, n_max_images=None):
    images = []
    with open(path, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = bytes_to_int(f.read(1))
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_mnist_labels(path, n_max_labels=None):
    labels = []
    with open(path, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels
    
def normalize_grayscale_image_data(image_data):
    # return [[(pixel / 255) for pixel in pixelrow] for pixelrow in image_data]
    return [[((pixel / 255 * 2) - 1) for pixel in pixelrow] for pixelrow in image_data]

def create_image_from_grayscale_data(data):
    image = Image.new(mode='LA', size=(len(data[0]), len(data)))
    for r, pixelrow in enumerate(data):
        for c, pixel in enumerate(pixelrow):
            image.putpixel((c, r), pixel)
    return image

def save_images_from_normalized_filtered_gsdata(images):
    for i, img in enumerate(images):
        img_data = [[(int((value + 1) * 128), 255) for value in valuerow] for valuerow in img[1]]
        create_image_from_grayscale_data(img_data).save(OUTPUTS + 'filtered_' + img[0] + '.png')

def read_features():
        return  [
                    read_image_data_in_grayscale_normalized(FEATURES + filename)
                    for filename    
                    in os.listdir(FEATURES)
                ]

def filter_and_pool_image(image, features, pool_constant=7):
    feature_filtered = get_feature_filtered_images(image, features)
    feature_filtered_and_pooled = []
    for fimage in feature_filtered:
        feature_filtered_and_pooled.append(pool_image(fimage, pool_constant))
    return feature_filtered_and_pooled

def get_feature_filtered_images(image, features):
    return [_filter_image_using_feature(image, feature) for feature in features]
    
def _filter_image_using_feature(image, feature):
    filtered_image = []
    for r in range(len(image)):
        filtered_image.append([])
        for c in range(len(image[r])):
            filtered_image[r].append(_get_resemblance(image, feature, r, c))
    return filtered_image
    
def _get_resemblance(image, feature, r_center, c_center):
    resemblance = 0
    for r_f in range(len(feature)):
        for c_f in range(len(feature[0])):
            r_i = r_center - int(len(feature) / 2) + r_f
            c_i = c_center - int(len(feature[0]) / 2) + c_f
            if(r_i >= 0 and r_i < len(image) and c_i >= 0 and c_i < len(image[0])):
                resemblance += feature[r_f][c_f] * image[r_i][c_i]
    return resemblance / (len(feature) * len(feature[0]))
  
def pool_image(image, window):
    pooled_image = [[-10 for r in range(math.ceil(len(image) / window))] 
                    for c in range(math.ceil(len(image[0]) / window))]
    r = 0
    for row in image:
        c = 0
        for value in row:
            r_b = math.floor(r/window)
            c_b = math.floor(c/window)
            pooled_image[r_b][c_b] = max(pooled_image[r_b][c_b], value)
            c+=1
        r+=1
    return pooled_image


def constrain_value(value, low, high):
    if(value < low):
        value = low
    elif(value > high):
        value = high
    return value
    
def flatten(matrix):
    return [value for valuelist in matrix for value in valuelist]

def unflatten(array, matrix_width):
    matrix = []
    i = 0
    for value in array:
        if(i == 0):
            matrix.append([])
        matrix[len(matrix) - 1].append(value)
        i+=1
        i%=matrix_width
    return matrix

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')