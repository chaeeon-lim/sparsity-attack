
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def check_nan(x, msg):
    if isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x)
    is_nan = tf.math.is_nan(x)
    # if is_nan
    has_nan = tf.reduce_any(is_nan)
    if has_nan:
        print(f"Tensor has NaN: {tf.get_static_value(has_nan)} ({msg})")
        
    return has_nan


def get_cifar10_dataset():

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize the images to [0, 1]
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Reshape the data to fit the model input
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    print(f"Training images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")

    return train_images, train_labels, test_images, test_labels



def get_mnist_dataset():

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize the images to [0, 1]
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Reshape the data to fit the model input
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    print(f"Training images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")

    return train_images, train_labels, test_images, test_labels

def get_dataset(nameDataset: str):
    if nameDataset == "cifar10":
        return get_cifar10_dataset()
    elif nameDataset == "mnist":
        return get_mnist_dataset()
    else:
        raise ValueError(f"Invalid dataset name: {nameDataset}")


def draw_decrease_in_activation_sparsity(betas, data):
    plt.figure(figsize=(4, 4))
    for name, d in data.items():
        plt.plot(betas, d, label=name, marker="o", markersize=4)
        
    plt.yticks([1, 1.2, 1.4, 1.6])
    plt.ylim((1, 1.8))
    plt.xlim((0, 20))
    plt.xticks([0, 10, 20])
    plt.axhline(y=1.2, color='gray', linewidth=0.5, zorder=0)
    plt.axhline(y=1.4, color='gray', linewidth=0.5, zorder=0)
    plt.axhline(y=1.6, color='gray', linewidth=0.5, zorder=0)

    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('\u03B2 \u2192') #beta , right arrow
    plt.ylabel('Dec. in Act. Sparsity')
    plt.legend()
    plt.savefig("output.png")
    plt.show()

def write_tensor_list(tensor_list, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for tensor in tensor_list:
            serialized_tensor = tf.io.serialize_tensor(tensor)
            writer.write(serialized_tensor.numpy())

def read_tensor_list(filename):
    records = tf.data.TFRecordDataset(filename)
    return [tf.io.parse_tensor(record, tf.float32) for record in records]