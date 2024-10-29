import tensorflow as tf
from tqdm import tqdm

# Define sigmoid activation function
def sigmoid(beta, x):
    return tf.keras.ops.sigmoid(beta*x)

# Define tanh activation function
def tanh(beta, x):
    return tf.keras.ops.tanh(beta*x)

def get_sparsity_function(name):
    if name.lower() == 'tanh':
        return tanh
    elif name.lower() == 'sigmoid':
        return sigmoid
    else:
        raise ValueError(f'Invalid function name: {name} (e.g. tanh, sigmoid)')


class SparsityModel():
    def __init__(self, model: tf.keras.models.Sequential, cin, cmax, cmin, Omax, Imax, mu, epsilon, epsilon_iter, sparsity_function, beta):
        self.model = model
        self.data_clean = []
        self.data_adverse = []
        self.cin = cin
        self.cmax = cmax
        self.cmin = cmin
        self.Omax = Omax
        self.Imax = Imax
        self.mu = mu
        self.epsilon = epsilon
        self.epsilon_iter = epsilon_iter
        self.sparsity_function = get_sparsity_function(sparsity_function)
        self.beta = beta
        
    def get_activation_sparsity(self, clean_data):
        activation_layers = [layer.output for layer in self.model.layers if 're_lu' in layer.name]
        
        # Get activation values for input
        activation_model = tf.keras.Model(inputs=self.model.inputs, outputs=activation_layers)
        activations = activation_model(clean_data)

        # Apply function to each activation
        sparsity_function_applied = map(lambda x: self.sparsity_function(self.beta, x), activations)
        sum_act_by_layers = list(map(lambda x: tf.reduce_sum(x, axis=[1,2,3]), sparsity_function_applied))
        estimate = tf.reduce_sum(sum_act_by_layers, axis=0)
        num_neurons = tf.reduce_sum(list(map(lambda x: x.shape[1:].num_elements(), activations)))
        num_neurons = tf.cast(num_neurons, dtype=tf.float32)

        activation_sparsity = -1 * estimate / num_neurons
        
        return activation_sparsity

    def get_adverse_input(self):
        return self.data_adverse
        # Get activation outputs

    def get_cross_entropy(self, pre_softmax_x_adv, pre_softmax_x_clean):
        y = tf.math.argmax(pre_softmax_x_clean, axis=-1, output_type=tf.int32)
        batch_indices = tf.range(y.shape[0])[:, tf.newaxis]
        argmax_indices = tf.expand_dims(y, axis=-1)
        coordinates = tf.concat([batch_indices, argmax_indices], axis=-1)

        numerator = tf.math.exp(tf.gather_nd(pre_softmax_x_adv, coordinates))
        denominator = tf.reduce_sum(tf.math.exp(pre_softmax_x_adv), axis=-1)
        cross_entropy = numerator / denominator
        return -1 * tf.math.log(cross_entropy)

#Algorithm 1
# x_clean(clean input), f(DNN model), L_sparsity and L_ce (Objective function terms)
# epsilon (Maximum L2 distortion), epsilon_iter (L2 distortion per iteration),
# O_max and I_max (Maximum outer and inner-loop iterations),
# c_in, c_min and c_max (Initial, min, and max value of trade-off constant)

    def algorithm_1(self, x_clean):
        # Get pre softmax values for input
        pre_softmax_out = self.model.layers[-1].output
        model_prime = tf.keras.Model(inputs=self.model.inputs, outputs=pre_softmax_out)
        y_clean = self.model(x_clean)
        c, o, x = tf.constant(self.cin, shape=y_clean.shape[0]), 0, 0

        while o < self.Omax:
            x = tf.convert_to_tensor(x_clean)
            i, g = 0, 0
            while i < self.Imax:
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    prediction = model_prime(x)
                    Lsparsity = self.get_activation_sparsity(x)
                    Lce = self.get_cross_entropy(prediction, y_clean)
                    loss = Lsparsity + c * Lce
                gradient = tape.gradient(loss, x)
                g  = self.mu*g + gradient
                x += -1*self.epsilon_iter*g/tf.norm(g)
                x  = tf.clip_by_value(x, 0, 1) #, epsilon, x_clean)
                i += 1

            y_clean_prime = tf.math.argmax(y_clean, axis=-1)
            y_adv_prime = tf.math.argmax(self.model(x), axis=-1)
            condition = tf.equal(y_clean_prime, y_adv_prime)
            c = tf.where(condition, (c+self.cmin)/2, (c+self.cmax)/2)
            o += 1

        return x

    def get_decrease_in_activation_sparsity(self, xClean):
        x_adv = self.algorithm_1(xClean)
        # for beta in tqdm(selbetas):
        sp_clean = self.get_activation_sparsity(xClean)
        sp_adv = self.get_activation_sparsity(x_adv)
        ratio = sp_adv/sp_clean
        if tf.math.reduce_any(tf.math.is_nan(ratio)):
            ratio = tf.where(tf.math.is_nan(ratio), tf.ones_like(ratio), ratio)
                
        return ratio