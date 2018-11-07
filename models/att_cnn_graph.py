from theano import tensor as T
import theano
import numpy as np
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams
srng2 = RandomStreams(seed=234)

from utils import *

class CNN(object):
    ''' CNN Model (http://www.aclweb.org/anthology/D14-1181)
    '''
    def __init__(self, emb, laplace, label_mat, B, nf=300, nc=2, de=300, p_drop=0.5, fs=[3,4,5], penalty=0,
            lr=0.001, decay=0., clip=None, train_emb=True):
        ''' Init Experimental CNN model.

            Args:
            emb: Word embeddings matrix (num_words x word_dimension)
            nc: Number of classes
            de: Dimensionality of word embeddings
            p_drop: Dropout probability
            fs: Convolution filter width sizes
            penalty: l2 regularization param
            lr: Initial learning rate
            decay: Learning rate decay parameter
            clip: Gradient Clipping parameter (None == don't clip)
            train_emb: Boolean if the embeddings should be trained
        '''
        fs = [10]
        nf = 50 #50
        self.emb = theano.shared(name='Words',
            value=as_floatX(emb))
        self.adj = theano.shared(name='Adj',
            value=as_floatX(laplace))

        self.filter_w = []
        self.filter_b = []
        for filter_size in fs:
            self.filter_w.append(theano.shared(
                value=he_normal((nf, 1, filter_size, de))
                .astype('float32')))
            self.filter_b.append(theano.shared(
                value=np.zeros((nf,)).astype('float32')))

        #self.w_count = theano.shared(value=he_normal((nf*len(fs)*nc, 1)).astype('float32'))
        #self.b_count = theano.shared(value=as_floatX(np.zeros((1,))))


        #self.b_o = theano.shared(value=as_floatX(np.zeros((nc,))))
        #self.w_o = theano.shared(value=he_normal((nf*len(fs), nc)).astype('float32'))
        nc = label_mat.shape[0]
        self.b_o = theano.shared(value=as_floatX(np.zeros((nc,))))
        self.w_o = theano.shared(value=he_normal((nc, nf*len(fs))).astype('float32'))
        self.w_o2 = theano.shared(value=he_normal((nc, nf*len(fs))).astype('float32'))

        self.w_h = theano.shared(value=he_normal((nf*len(fs), 128)).astype('float32'))
        self.b_h = theano.shared(value=as_floatX(np.zeros((128,))))
        self.w_att = theano.shared(value=he_normal((128, nc)).astype('float32'))
        hid_s = 256
        self.w_g1 = theano.shared(value=he_normal((nf*len(fs)*2, hid_s)).astype('float32'))
        self.w_b1 = theano.shared(value=he_normal((hid_s,)).astype('float32'))
        self.w_g2 = theano.shared(value=he_normal((hid_s, nf*len(fs))).astype('float32'))
        self.w_b2 = theano.shared(value=he_normal((nf*len(fs),)).astype('float32'))

        #self.w_o = theano.shared(value=he_normal((nh2, nf*len(fs))).astype('float32'))

        self.params = [self.emb, self.w_o, self.w_o2, self.b_o, self.w_h, self.b_h, self.w_att]#, self.w_out]
        #self.params_counts = [self.w_count, self.b_count]

        for w, b in zip(self.filter_w, self.filter_b):
            self.params.append(w)
            self.params.append(b)
        self.params += [self.w_g1, self.w_g2, self.w_b1, self.w_b2]

        dropout_switch = T.fscalar('dropout_switch')

        idxs = T.matrix()
        x_word = self.emb[T.cast(idxs.flatten(), 'int32')].reshape((idxs.shape[0], 1, idxs.shape[1], de))
        x_word = dropout(x_word, dropout_switch, 0.2)
        mask = T.neq(idxs, 0)*as_floatX(1.)
        x_word = x_word*mask.dimshuffle(0, 'x', 1, 'x')
        Y = T.matrix('Y')
        Y_counts = T.vector('Yc')

        w = self.filter_w[0]
        b = self.filter_b[0]
        width = fs[0]
        l1_w = conv2d(x_word, w, input_shape=(None,1,None,de), filter_shape=(nf, 1, width, de))
        l1_w = T.nnet.relu(l1_w + b.dimshuffle('x', 0, 'x', 'x'))
        l1_w = l1_w.flatten(3).dimshuffle(0,2,1)
        #l1_w = T.max(l1_w, axis=2).flatten(2)
        l1_w_att = T.tanh(T.dot(l1_w, self.w_h) + self.b_h)
        l1_w_att = T.dot(l1_w_att, self.w_att)
        #l1_w_att = T.dot(l1_w_att, w_graph)

        # batch_size x num_words-4+1 x nc
        #l1_w_att.dimshuffle(0,2,1)
        #l1_w = l1_w.dimshuffle(2,0,1)
        #(20, 3832, 50)
        #(20, 3832, 7042)

        # batch_size:vs 
        def attention_rec(ht, hat2, wo):
            # hat2 = num_words x num_classes
            hat = T.nnet.nnet.softmax(hat2.T)
            # hat = num_classes x num_words
            # ht = num_words x num_filters
            # returns num_classes x num_filters*2
            tmp = hat.dot(ht)
            tmp2 = T.concatenate([tmp, wo], axis=1)
            return tmp, tmp2

        # batch_size x nc x num_filters*2
        [h2, h], updates = theano.scan(attention_rec, sequences=[l1_w, l1_w_att],
                non_sequences=[self.w_o2],
                outputs_info=[None, None])

        h = h.dimshuffle(0,2,1)
        pyx1 = T.nnet.sigmoid((h2 * self.w_o.dimshuffle('x',0,1)).sum(axis=2) + self.b_o)
        #h = dropout(h, dropout_switch, 0.5)
        #w_graph = dropout(w_graph, dropout_switch, 0.2)
        w_graph = T.dot(h, self.adj).dimshuffle(0,2,1)
        w_graph = T.nnet.relu(T.dot(w_graph, self.w_g1) + self.w_b1).dimshuffle(0,2,1)
        w_graph = T.dot(w_graph, self.adj).dimshuffle(0,2,1)
        w_graph = T.dot(w_graph, self.w_g2) + self.w_b2
        #w_graph = dropout(w_graph, dropout_switch, 0.2)
        pyx = T.nnet.sigmoid((w_graph * self.w_o.dimshuffle('x',0,1)).sum(axis=2) + self.b_o)
        #pyx = T.nnet.sigmoid((h * self.w_o.dimshuffle('x',1,0)).sum(axis=2) + self.b_o)
        #pyx = T.nnet.sigmoid(w_graph.dot(self.w_out) + self.b_o)
        pyx = T.clip(pyx, 1e-7, 1.-1e-7)
        loss = dropout2(T.nnet.nnet.binary_crossentropy(pyx[:,:7042], Y[:,:7042]), dropout_switch, 0.2)
        L = loss.sum() + 1e-4 * sum([(x**2).sum() for x in self.params[:-4]])

        #count = T.nnet.relu(T.dot(h.flatten(2), self.w_count) + self.b_count).flatten()
        #L_count = ((count - Y_counts)**2).sum()
        updates = Adam(L, self.params, lr2=lr, clip=clip)
        #updates_count = Adam(L_count, self.params_counts, lr2=lr, clip=clip)

        self.mid_feat = theano.function([idxs, dropout_switch], [l1_w, l1_w_att], allow_input_downcast=True, on_unused_input='ignore')
        self.train_batch = theano.function([idxs, Y, dropout_switch], [L, pyx, h], updates=updates, allow_input_downcast=True, on_unused_input='ignore')
        #self.train_count = theano.function([idxs, Y_counts, dropout_switch], L_count, updates=updates_count, allow_input_downcast=True, on_unused_input='ignore')

        self.predict = theano.function([idxs, dropout_switch],
                outputs=pyx, allow_input_downcast=True, on_unused_input='ignore')
        self.predict_proba = theano.function([idxs, dropout_switch], outputs=pyx, allow_input_downcast=True, on_unused_input='ignore')
        self.predict_loss = theano.function([idxs, Y, dropout_switch], [pyx, L], allow_input_downcast=True, on_unused_input='ignore')

    def __getstate__(self):
        return [x.get_value() for x in self.params]# + [x.get_value() for x in self.params_counts]

    def __setstate__(self, data):
        for w, p in zip(data, self.params):#+self.params_counts):
            p.set_value(w)
        return

    def __setstate2__(self, data):
        self.emb.set_value(data[0])
        #self.w_o.set_value(data[1])
        self.b_o.set_value(data[2])
        cnt = 3
        for f1 in self.filter_w:
            f1.set_value(data[cnt])
            cnt += 1
        for f1 in self.filter_b:
            f1.set_value(data[cnt])
            cnt += 1

