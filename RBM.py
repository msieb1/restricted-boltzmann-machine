import math
import random
import numpy as np
import csv
import re
import matplotlib.pyplot as plt


class RBM:
    """
    create a 1-layer RBM as class object
    """
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True
        self.W = np.zeros((num_hidden, num_visible))
        self.b = np.zeros((num_hidden, 1))
        self.c = np.zeros((num_visible, 1))
        # Initialize parameters
        row = num_hidden
        col = num_visible
        np.random.seed()
        self.W = np.reshape(np.random.normal(0, 0.1, row * col), [row, col])
        self.b = np.reshape(np.random.normal(0, 0.1, row ), [row, 1])
        self.c = np.reshape(np.random.normal(0, 0.1, col), [col, 1])

    def sigmoid(self, a):
        np.clip(a,-300,300)
        return 1/(1+np.exp(-a))

    def gibbs_sampler(self, visible_states, steps):
        """
        Standard gibbs sampler
        :param visible_states: initial starting point - p(h|v) is sampled first
        :param steps:
        :return:
        """
        num_input = visible_states.shape[1]
        num_visible = visible_states.shape[0]
        sigm = self.sigmoid
        for k in np.arange(0, steps):
            hidden_act = np.dot(self.W, visible_states) + self.b
            hidden_probs = sigm(hidden_act)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden, num_input)

            visible_act = np.dot(self.W.T, hidden_states) + self.c
            visible_probs = sigm(visible_act)
            visible_states = visible_probs > np.random.rand(self.num_visible, num_input)
        return hidden_states, visible_states, hidden_probs, visible_probs

    def train(self, data, n_epochs = 100, l_rate = 0.1, batch_size = 20, gibbs_steps = 1):
        """
        Train the RBM
        :param x_train: data to be trained upon
        :param n_epochs: number of training epochs
        :param l_rate: learning rate
        :param batch_size: 1 equals SGD
        :param gibbs_steps: how many gibbs steps for the Contrastive Divergence update
        :return:
        """
        x_train = data['x_train']
        x_valid = data['x_valid']
        num_samples = x_train.shape[1]
        dim_inputs = x_train.shape[0]
        sigm = self.sigmoid

        error_tracker = {'train': np.zeros((n_epochs + 1, 1)), 'valid': np.zeros((n_epochs + 1, 1))}

        for epoch in range (n_epochs):

            # Randomize input data for every epoch
            np.random.seed()
            randomize = np.arange(x_train.shape[1])
            np.random.shuffle(randomize)
            x_train = x_train[:, randomize][:, :]

            for itr in np.arange(int(num_samples/batch_size)):
                batch_data = x_train[:, itr * batch_size:(itr + 1) * batch_size]  #fix index passing
                # Generate sample of visible units
                sampled_hidden_states, sampled_visible_states, hidden_probs, visible_probs = self.gibbs_sampler(batch_data, gibbs_steps)

                # Update W
                # Positive Phase
                pos_hidden_act = np.dot(self.W, batch_data) + self.b
                pos_hidden_probs = sigm(pos_hidden_act)
                pos_phase = np.dot(pos_hidden_probs, batch_data.T)
                # Negative Phase
                neg_hidden_act = np.dot(self.W, sampled_visible_states) + self.b
                neg_hidden_probs = sigm(neg_hidden_act)
                neg_phase = np.dot(hidden_probs, visible_probs.T)
                self.W += l_rate*(pos_phase - neg_phase)/batch_size

                # Update b
                # Positive Phase
                pos_phase = pos_hidden_probs
                # Negative Phase
                neg_phase = hidden_probs
                self.b += l_rate*np.sum(pos_phase - neg_phase)/batch_size

                # Update c
                # Positive Phase
                pos_phase = batch_data
                # Negative Phase
                neg_phase = visible_probs
                self.c += l_rate*np.sum(pos_phase - neg_phase)/batch_size

            # Calculate Cross Entropy Error for every epoch
            error_valid = self.cross_entropy_error(x_valid)
            error_tracker['valid'][epoch + 1, 0] = error_valid
            error = self.cross_entropy_error(x_train)
            error_tracker['train'][epoch + 1, 0] = error

            hidden_act = np.dot(self.W, x_train) + self.b
            hidden_probs = self.sigmoid(hidden_act)
            visible_act = np.dot(self.W.T, hidden_probs) + self.c
            visible_probs = self.sigmoid(visible_act)
            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch+1, error))
                #print("Reconstruction error: %s" % (np.sum(np.abs(x_train - visible_probs))))

            #sampled_hidden_states, sampled_visible_states = self.gibbs_sampler(x_valid, 1)
            if epoch%1 == 0:
                l_rate /= 1.02
        np.save('W_RBM100.npy', self.W)

        return error_tracker

    def cross_entropy_error(self, data):
        eps = 1e-10
        try:
            dim, n = data.shape
        except:
            dim = len(data)
            n = 1
        sigm = self.sigmoid
        loss = 0
        weight_loss = 0
        hidden_act = np.dot(self.W, data) + self.b
        hidden_probs = sigm(hidden_act)
        visible_act = np.dot(self.W.T, hidden_probs) + self.c
        visible_probs = sigm(visible_act)
        # Take expectation of h and then expectation of x and then plug into cross entropy
        loss = -data*np.log(visible_probs+eps) - (1-data)*np.log(1-visible_probs+eps)
        loss = np.sum(loss)/n
        return loss

    def sample_image(self, data, gibbs_chain = 100, sample_steps = 1000):
        np.random.seed()
        #init = np.reshape(np.random.normal(0, 0.1, self.num_visible*gibbs_chain), [self.num_visible, gibbs_chain])
        init = np.reshape(np.random.rand(self.num_visible*gibbs_chain), [self.num_visible, gibbs_chain])
        images = np.zeros((self.num_visible, gibbs_chain))
        sampled_hidden_states, sampled_visible_states, _ ,_ = self.gibbs_sampler(init, sample_steps)
        images = sampled_visible_states.T
        return images



def plotting(path_to_save, images, error_tracker, save_to_file=False):
    n = '500'
    # 1. Plot feature map W
    path_to_file = '/home/max/PyCharm/PycharmProjects/10-707/hw2/'
    save = save_to_file
    W = np.load(path_to_file + 'W.npy')
    l = W.shape[0]
    W = W.reshape(l, 28, 28)
    r = int(math.sqrt(l))
    out = np.zeros([28 * r, 28 * r])
    for i in range(r):
        for j in range(r):
            out[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = W[i * r + j]
    plt.figure(1)
    plt.imshow(out, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    name = 'Features_' + n
    path = path_to_save + name
    if save:
        plt.savefig(path)

    # 2. Plot sampled images
    l = images.shape[0]
    W = images.reshape(l, 28, 28)
    r = int(math.sqrt(l))
    out = np.zeros([28 * r, 28 * r])
    for i in range(r):
        for j in range(r):
            out[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = W[i * r + j]
    plt.figure(2)
    plt.imshow(out, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    name = 'SampledImages_' + n
    path = path_to_save + name
    if save:
        plt.savefig(path)

    # 3. Plot error
    plt.figure(3)
    plt.clf()
    plt.plot(np.arange(1, len(error_tracker['train'])),error_tracker['train'][1:], 'bo-', markersize=4, label='train')
    plt.plot(np.arange(1, len(error_tracker['valid'])),error_tracker['valid'][1:], 'go-', markersize=4, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(loc='upper left')
    name = 'CrossEntropyError_' + n
    path = path_to_save + name
    if save:
        plt.savefig(path)
    #plt.show()

def load_data():
    with open('/home/max/PyCharm/PycharmProjects/10-707/hw2/data/digitstrain.txt') as f:
        reader = csv.reader(f, delimiter=" ")
        d = list(reader)
        x_train = np.zeros((784,3000))
        y_train = np.zeros((10,3000))
        for i in range(0, len(d)):
            s = d[i][0]
            p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
            data = [float(i) for i in p.findall(s)]  # Convert strings to float
            p = re.compile(r'\d+')  # Compile a pattern to capture float values
            label = [int(i) for i in p.findall(s)]
            x_train[:,i] = data
            y_train[label[-1],i] = 1
            np.save('x_train.npy', x_train)
            np.save('y_train.npy', y_train)

    with open('/home/max/PyCharm/PycharmProjects/10-707/hw2/data/digitstest.txt') as f:
        reader = csv.reader(f, delimiter=" ")
        d = list(reader)
        x_test = np.zeros((784, 3000))
        y_test = np.zeros((10, 3000))
        for i in range(0, len(d)):
            s = d[i][0]
            p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
            data = [float(i) for i in p.findall(s)]  # Convert strings to float
            p = re.compile(r'\d+')  # Compile a pattern to capture float values
            label = [int(i) for i in p.findall(s)]
            x_test[:, i] = data
            y_test[label[-1], i] = 1
            np.save('x_test.npy', x_test)
            np.save('y_test.npy', y_test)

    with open('/home/max/PyCharm/PycharmProjects/10-707/hw2/data/digitsvalid.txt') as f:
        reader = csv.reader(f, delimiter=" ")
        d = list(reader)
        x_valid = np.zeros((784, 1000))
        y_valid = np.zeros((10, 1000))
        for i in range(0, len(d)):
            s = d[i][0]
            p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
            data = [float(i) for i in p.findall(s)]  # Convert strings to float
            p = re.compile(r'\d+')  # Compile a pattern to capture float values
            label = [int(i) for i in p.findall(s)]
            x_valid[:, i] = data
            y_valid[label[-1], i] = 1
            np.save('x_valid.npy', x_valid)
            np.save('y_valid.npy', y_valid)
    dat = {'x_train': x_train}
    dat['y_train'] = y_train
    dat['x_test'] = x_test
    dat['y_test'] = y_test
    dat['x_valid'] = x_valid
    dat['y_valid'] = y_valid

    return dat
# ONLY LOAD IF .npy files not stored!!
data = load_data()

#NORMALIZING?????????????????????????????????
x_train = (np.load('x_train.npy'))
y_train = (np.load('y_train.npy'))
x_valid = (np.load('x_valid.npy'))
y_valid = (np.load('y_valid.npy'))
x_test = (np.load('x_test.npy'))
y_test = (np.load('y_test.npy'))
# store data in a dictionary
data = {'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid, 'x_test': x_test, 'y_valid': y_valid}
dim_inputs = x_train.shape[0]


##### SET HYPERPARAMETERS AND UNIT CONFIGURATON ########
num_hidden = 100
l_rate = 0.05
batch_size = 50
n_epochs = 100
gibbs_steps = 20
################################################

r = RBM(num_visible=dim_inputs, num_hidden=num_hidden)
error_tracker = r.train(data=data, n_epochs=n_epochs, l_rate=l_rate, batch_size=batch_size, gibbs_steps=gibbs_steps)
images = r.sample_image(x_train,100,1000)
path = '/home/max/PyCharm/PycharmProjects/10-707/hw2/figures/'
plotting(path, images, error_tracker, save_to_file=False)
plt.show()

