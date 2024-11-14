import tensorflow as tf
from rich.console import Console
from utils import write_tensor_list, read_tensor_list

console = Console()

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

class SparsityModel(tf.keras.Model):
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
        # print(f"estimate: {estimate} ({estimate.shape})")
        num_neurons = tf.reduce_sum(list(map(lambda x: x.shape[1:].num_elements(), activations)))
        # print(f"num_neurons: {num_neurons} ({num_neurons.shape})")
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
        cross_entropy = numerator/ denominator
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
        # print(f"[DEBUG] beta: {self.beta}")
        while o < self.Omax:
            x = tf.identity(x_clean)
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

                console.print(f"loss: {tf.reduce_mean(loss)}, gradient: {tf.reduce_mean(gradient)}, g: {tf.reduce_mean(g)}")

                x = x - self.epsilon_iter*g/tf.norm(g, ord=2)
                x  = tf.clip_by_value(x, 0, 1)
                i += 1

            y_clean_prime = tf.math.argmax(y_clean, axis=-1)
            y_adv_prime = tf.math.argmax(self.model(x), axis=-1)
            condition = tf.equal(y_clean_prime, y_adv_prime)
            c = tf.where(condition, (c+self.cmin)/2, (c+self.cmax)/2)
            o += 1

        return x

    def get_decrease_in_activation_sparsity(self, xClean):
        x_adv = self.algorithm_1(xClean)
        sp_clean = self.get_activation_sparsity(xClean)
        sp_adv = self.get_activation_sparsity(x_adv)
        ratio = sp_adv/sp_clean
        if tf.math.reduce_any(tf.math.is_nan(ratio)):
            ratio = tf.where(tf.math.is_nan(ratio), tf.ones_like(ratio), ratio)
                
        return ratio
    
    def eval_decrese_in_activation_sparsity_by_func(self, func, betas, xClean):
        funcName = func.__name__
        with console.status(f"[bold green] evaluating with function: [italic red]{funcName}[bold green]") as status:
            tmp = []
            for i, beta in enumerate(betas):
                status.update(f"[bold green] evaluating with function: [italic red]{funcName}[/italic red][bold green] (beta: {beta}, {(i+1)/len(betas)*100:.1f}%)")
                self.beta = beta
                self.sparsity_function = func
                ratio = self.get_decrease_in_activation_sparsity(xClean)
                tmp.append(ratio)

        result = tf.stack(tmp, axis=1)
        result = tf.reduce_mean(result, axis=0)
        return result

    
    def eval_decrease_in_activation_sparsity(self, evalConfig):
        def parse_range(givenRange):
            if type(givenRange) is int:
                return givenRange, givenRange+1
            else:
                result = givenRange.split(sep=":", maxsplit=1)
                if len(result) == 2:
                    return list(map(int, result))
                else:
                    return result
        result = {}
        targetIndexStart, targetIndexEnd = parse_range(evalConfig['testDataRange']) if 'testDataRange' in evalConfig.keys() else [0,1]
        betaOrig = int(self.beta)
        funcOrig = self.sparsity_function
        if "betas" in evalConfig.keys():
            betas = evalConfig['betas']
        else: 
            betas = [self.beta]
        
        if 'functions' in evalConfig.keys():
            if type(evalConfig["functions"]) is str:
                funcs = [get_sparsity_function(evalConfig["functions"])]
            elif type(evalConfig["functions"]) is list:
                funcs = [get_sparsity_function(f) for f in evalConfig["functions"]]
        else:
            funcs = [self.sparsity_function]
        if 'useSaved' in evalConfig.keys():
            for func, path in zip(funcs, evalConfig['useSaved']):
                funcName = func.__name__
                console.log(f"Use save data: {funcName} <- {path}")
                result[funcName] = read_tensor_list(path)
        else:
            for func in funcs:
                funcName = func.__name__
                console.log(f"Evaluate data[{targetIndexStart}:{targetIndexEnd}] with {funcName} function")
                result[funcName] = self.eval_decrese_in_activation_sparsity_by_func(func, betas, self.data_clean[targetIndexStart:targetIndexEnd])
                console.log(f"func: {funcName} completed (ratio: {result[funcName]})")
                if 'saveName' in evalConfig.keys():
                    path = evalConfig['savePath'] if 'savePath' in evalConfig.keys() else "outputs/"
                    write_tensor_list(result[funcName], f"{path}/{evalConfig['saveName']}_{funcName}.tfrecrod")
            self.beta = betaOrig
            self.sparsity_function = funcOrig
        return result