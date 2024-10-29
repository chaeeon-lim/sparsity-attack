from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def compile_model(func):
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)  # Call the original __init__ method
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    return wrapper

@compile_model
def Model1():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

@compile_model
def Model2():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), padding="same", input_shape=(28, 28, 1), activation='relu'),
        layers.MaxPooling2D((3, 3)),
        layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
        layers.MaxPooling2D((3, 3)),
        layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

@compile_model
def Model3():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

@compile_model
def Mnist_Conv():
    model = models.Sequential([
        layers.Conv2D(20, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        layers.Conv2D(20, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(20, (3, 3), activation='relu'),
        layers.Conv2D(20, (3, 3), activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

@compile_model
def Cifar10_Conv1():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(32, 32, 1), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def Cifar10_Conv2():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Dropout(0.2),
        layers.Conv2D(96,  (3, 3), padding="same", kernel_initializer='glorot_uniform'),
        layers.ReLU(),
        layers.Conv2D(96,  (3, 3), padding="same", kernel_initializer='glorot_uniform'),
        layers.ReLU(),
        layers.Conv2D(96,  (3, 3), padding="same", kernel_initializer='glorot_uniform', strides=2),
        layers.ReLU(),
        layers.Dropout(0.5),
        
        layers.Conv2D(192, (3, 3), padding="same", kernel_initializer='glorot_uniform'),
        layers.ReLU(),
        layers.Conv2D(192, (3, 3), padding="same", kernel_initializer='glorot_uniform'),
        layers.ReLU(),
        layers.Conv2D(192, (3, 3), padding="same", kernel_initializer='glorot_uniform', strides=2),
        layers.ReLU(),
        layers.Dropout(0.5),

        layers.Conv2D(192, (3, 3), padding="valid", kernel_initializer='glorot_uniform'),
        layers.ReLU(),
        layers.Conv2D(192, (1, 1), kernel_initializer='glorot_uniform'),
        layers.ReLU(),
        layers.Conv2D(10,  (1, 1)),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Softmax(),
    ])
    optimizer = Adam(learning_rate=0.05, weight_decay=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model(modelName: str):
    modelName = modelName.lower()
    if modelName == 'model1':
        return Model1()
    elif modelName == 'model2':
        return Model2()
    elif modelName == 'model3':
        return Model3()
    elif modelName == 'mnist_conv':
        return Mnist_Conv()
    elif modelName == 'cifar10_conv1':
        return Cifar10_Conv1()
    elif modelName == 'cifar10_conv2':
        return Cifar10_Conv2()
    else:
        raise ValueError(f'modelName is not supported: {modelName}')
    
