import random
import numpy as np
import pickle

class Network:
    def __init__(self, sizes, cost_func="MSE"):
        self.cost_func = cost_func #options: "cross_entropy", "MSE", "soft_max"
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
        #biases is a list of matrices (each with dimension num_nodes_in_layer by 1
        x,y = sizes[0], sizes[1]
        self.weights = [np.random.normal(loc=0.0, scale=1/np.sqrt(x), size=(y,x))]
        for x,y in zip(sizes[1:-1], sizes[2:]):
            self.weights.append(np.random.randn(y,x))
        #weights is a list of matrices (each with dimension num_nodes_in_dest_layer by num_nodes_in_src_layer)

    #return network output for input a
    def feedforward(self, a):
        for b,w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w,a) + b)
               
        b = self.biases[-1]
        w = self.weights[-1]
        if self.cost_func == "soft_max":
            a = softmax(np.dot(w,a) + b)
        else:
            a = sigmoid(np.dot(w,a) + b)

        return a

    #training data is a list of (x,y) tuples
    def SGD(self, training_data, mini_batch_size, eta, validation_data=None, regularization_param = 0.0, epochs_to_decrease_eta=10, eta_decrease_fraction=0.5, num_eta_decreases=5):
        n = len(training_data)
        eta_decrease_count=0
        num_stagnated = 0
        max_num_correct = None
        j = 0
        while eta_decrease_count <= num_eta_decreases:
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_from_mini_batch(mini_batch, eta, regularization_param, len(training_data))

            num_correct = self.evaluate(validation_data)
            print("epoch {0}: {1} / {2}".format(j, num_correct, len(validation_data)))

            if num_correct == len(validation_data):
                print("got em all right! saving network")
                return

            if j == 0:
                max_num_correct = num_correct
            else:
                if num_correct <= max_num_correct:
                    num_stagnated += 1
                else:
                    max_num_correct = num_correct

            if num_stagnated == epochs_to_decrease_eta:
                eta *= eta_decrease_fraction
                print("decreasing eta: eta now ", eta)
                eta_decrease_count+=1
                num_stagnated = 0
            j+=1
        
    def update_from_mini_batch(self, mini_batch, eta, regularization_param, n):
        b_gradient_sums = [np.zeros(b.shape) for b in self.biases]
        w_gradient_sums = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            b_gradients, w_gradients = self.backprop(x,y,regularization_param, n)
            b_gradient_sums = [grad_sum + grad for grad_sum, grad in zip(b_gradient_sums, b_gradients)]
            w_gradient_sums = [grad_sum + grad for grad_sum, grad in zip(w_gradient_sums, w_gradients)]
        b_gradient_avgs = [grad_sum / len(mini_batch) for grad_sum in b_gradient_sums]
        w_gradient_avgs = [grad_sum / len(mini_batch) for grad_sum in w_gradient_sums]
        
        #w_gradient_avgs = [grad_avg + regularization_param * w / n for grad_avg, w in zip(w_gradient_avgs, self.weights)]

        self.weights = [weight - eta * w_gradient_avg for weight, w_gradient_avg in zip(self.weights, w_gradient_avgs)]

        self.biases = [bias - eta * b_gradient_avg for bias, b_gradient_avg in zip(self.biases, b_gradient_avgs)]
    

    def backprop(self, x, y, regularization_param, n):
        """Return a tuple ``(b_gradients, w_gradients)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        b_gradients = [np.zeros(b.shape) for b in self.biases]
        w_gradients = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        if self.cost_func == "MLP":
            delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        elif self.cost_func == "cross_entropy":
            delta = (activations[-1] - y)
        elif self.cost_func == "soft_max":
            delta = (activations[-1] - y)
        b_gradients[-1] = delta
        w_gradients[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            b_gradients[-l] = delta
            w_gradients[-l] = np.dot(delta, activations[-l-1].transpose())

        if regularization_param != None:
            w_gradients = [w_grad + (w * regularization_param / n) for w_grad,w in zip(w_gradients, self.weights)]
        
        return (b_gradients, w_gradients)

    def evaluate(self, validation_data):
        total = 0
        for x,y in validation_data:
            if np.argmax(self.feedforward(x)) == np.argmax(y):
                total += 1
        return total
            
		

#Misc. Functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    total = sum(np.exp(z))
    return np.exp(z)/total

    
    
