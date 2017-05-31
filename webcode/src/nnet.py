import random
import numpy as np


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class Network(object):

    def __init__(self, layers, cost_function=QuadraticCost):
        self.num_layers = len(layers)
        #layers is a python list of num_nodes per layer in NN
        self.layers = layers
        #biases is a python list of numpy arrays size num_nodes x 1 for each non input layer
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
        #weights is a python list of numpy arrays size num_nodes(l) x num_nodes(l-1) for each l, l-1 in NN
        self.init_weights(layers)
        self.cost=cost_function

    def init_weights(self,layers):
        #modify this to the advanced method
        self.weights = [np.random.rand(y,x)/np.sqrt(x) for x,y in zip(layers[:-1],layers[1:])]

    #takes input, dot product through the NN returning a numpy array num_nodes(L) x 1 of output(s)
    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
            #alternate implementation(s)
            #a = hypertan(np.dot(w,a)+b)
        return a


    def SGD(self, training_data, epochs, batch_size, eta, lmbda = 0.0,
            eval_data=None,
            monitor_eval_cost=False,
            monitor_eval_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if eval_data: n_data = len(eval_data)
        n = len(training_data)
        eval_cost, eval_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in xrange(epochs):
            random.shuffle(training_data)
            samples = [ training_data[k:k+batch_size] for k in xrange(0,n,batch_size) ]
            print len(samples)
            for sample in samples:
                self.update_samples(sample, eta, lmbda, n)
            print "epoch %s training complete on %s sets" % (j,len(samples))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_eval_cost:
                cost = self.total_cost(eval_data, lmbda, convert=True)
                eval_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_eval_accuracy:
                accuracy = self.accuracy(eval_data)
                eval_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(self.accuracy(eval_data), n_data)
        return eval_cost, eval_accuracy, training_cost, training_accuracy


    def update_samples(self, sample, eta, lmbda, n):
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        for x, y in sample:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [ nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b) ]
            nabla_w = [ nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w) ]
        self.weights = [ (1-eta*(lmbda/len(sample)))*w-(eta/len(sample))*nw for w, nw in zip(self.weights, nabla_w) ]
        self.biases = [ b-(eta/len(sample))*nb for b, nb in zip(self.biases, nabla_b) ]


    def backprop(self, x, y):
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        activation = x
        # feedforward - same as standalone function
        #python list of activations by layer
        activations = [x]
        zvector = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zvector.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # back propogate
        #calculate delta(L)
        delta = (self.cost).delta(zvector[-1], activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #calculate delta(l's)
        for l in xrange(2, self.num_layers):
            z = zvector[-l]
            sigprime = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigprime
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()



def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


