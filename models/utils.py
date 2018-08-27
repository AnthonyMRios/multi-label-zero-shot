from theano import tensor as T
#import theano.sandbox.cuda
from collections import OrderedDict
from theano.ifelse import ifelse
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

np.random.seed(1234)
rng = np.random.RandomState(1234)
srng = RandomStreams(rng.randint(54321))

def general_softmax(x, axis=1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]
        #return [0. * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.mean(targets * log_predictions, axis=1)

def normal(shape, scale=0.05):
    return np.random.normal(0, scale, size=shape).astype('float32')

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def orthogonal(shape):
    ''' Reference: Glorot & Bengio, AISTATS 2010 glorot_normal
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in * fan_out))
    return normal(shape, s)

def he_normal(shape):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s)

def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)

def orthogonal_tmp2(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)

def uniform(shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')

def orthogonal_tmp(shape, scale=1.0):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast['float32'](variable)
    elif isinstance(variable, np.ndarray):
        return np.cast['float32'](variable)

def lrelu(X):
    return T.maximum(X, 0.01*X)

def rectify(X):
    return T.maximum(X, 0.)

def cappedrectify(X):
    return T.minimum(5., T.maximum(X, 0.))

def elu(X):
    return T.switch(T.ge(X, 0), X, T.exp(X)-1.)

def snelu(X):
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * T.switch(T.ge(X, 0), X, alpha*T.exp(X)-alpha)

def dropout(X, dropout_switch=1, p=0.):
    retain_prob = 1 - p
    mask = srng.binomial(X.shape, p=retain_prob, dtype='float32')
    X = ifelse(T.lt(dropout_switch, 0.5), X*mask, (X*(retain_prob)).reshape(mask.shape))
    return X

def dropout2(X, dropout_switch=1, p=0.):
    retain_prob = 1 - p
    mask = srng.binomial(X.shape, p=retain_prob, dtype='float32')
    X = ifelse(T.lt(dropout_switch, 0.5), X*mask, (X).reshape(mask.shape))
    return X


def dropout_scan(X, mask, dropout_switch=1, p=0.):
    retain_prob = 1 - p
    X = ifelse(T.lt(dropout_switch, 0.5), X*mask, (X*retain_prob).reshape(mask.shape))
    return X

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g * c / n, g)
    return g

def sgdm(cost, parameters, lr2=1., momentum=0.8):
    lr = theano.shared(as_floatX(lr2).astype("float32"))
    grads = T.grad(cost, parameters)
    updates = OrderedDict()
    for param,g2 in zip(parameters,grads):
        grad = clip_norm(g2, 3, T.sum(g2 ** 2))
        mparam = theano.shared(param.get_value()*0.)
        updates[param] = param - lr * mparam
        updates[mparam] = mparam*momentum + (1.-momentum)*grad

    return updates, lr

def sgd(cost, parameters, lr2, decay=0., updates=None):
    lr = theano.shared(as_floatX(lr2).astype("float32"))
    i = theano.shared(as_floatX(0.))
    grads = T.grad(cost,parameters)
    updates = OrderedDict({})
    for param,grad in zip(parameters,grads):
            updates[param] = param - lr*grad
    updates[lr] = lr*(1./(1.+decay*i))
    i_t = i + as_floatX(1.)
    updates[i] = i_t
    return updates

def Adam(cost, params, lr2=0.001, b1=0.1, b2=0.001, e=1e-8, decay=0., clip=None):
    clip = None
    updates = []
    lr = theano.shared(as_floatX(lr2).astype("float32"))
    grads = T.grad(cost, params)
    i = theano.shared(as_floatX(0.))
    i_t = i + as_floatX(1.)
    fix1 = as_floatX(1.) - (as_floatX(1.) - as_floatX(b1))**i_t
    fix2 = as_floatX(1.) - (as_floatX(1.) - as_floatX(b2))**i_t
    #lr_t = as_floatX(lr) * (T.sqrt(fix2) / fix1)
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g2 in zip(params, grads):
        if clip is not None and False:
            g = clip_norm(g2, clip, T.sum(g2 ** 2))
        else:
            g = g2
        m = theano.shared(p.get_value() * as_floatX(0.))
        v = theano.shared(p.get_value() * as_floatX(0.))
        m_t = (as_floatX(b1) * g) + ((as_floatX(1.) - as_floatX(b1)) * m)
        v_t = (as_floatX(b2) * T.sqr(g)) + ((as_floatX(1.) - as_floatX(b2)) * v)
        g_t = m_t / (T.sqrt(v_t) + as_floatX(e))
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    updates.append((lr, lr*(1./(1.+decay*i))))
    return lr, updates

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    '''
    norm = T.sqrt(sum([T.sum(g ** 2) for g,p in zip(grads, params) if p.name != 'Words' and p.get_value(borrow=True).ndim == 2 and p.name != 'label_embeddings']))
    tmp_grads = []
    for g,p in zip(grads, params):
        if p.name != 'Words' and (p.get_value(borrow=True).ndim == 2) and p.name != 'label_embeddings':
            tmp_grads.append(clip_norm(g, 5, norm))
        else:
            tmp_grads.append(g)
    grads = tmp_grads
    '''
    norm = T.sqrt(sum([T.sum(g ** as_floatX(2.)) for g in grads]))
    grads = [clip_norm(g, as_floatX(5.), norm) for g in grads]
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * as_floatX(0.))
        acc_new = as_floatX(rho) * acc + (as_floatX(1.) - as_floatX(rho)) * g ** as_floatX(2.)
        gradient_scaling = T.sqrt(acc_new + as_floatX(epsilon))
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - as_floatX(lr) * g))
    return updates


def adagrad(cost, params, lr=0.001, eps=1e-8, sparse=False):
    lr = theano.shared(as_floatX(lr).astype("float32"))
    eps = as_floatX(eps).astype("float32")

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))+0.1) for param in params]
    #gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]
    xsums   = [None for param in params]

    gparams = T.grad(cost, params)

    updates = OrderedDict()

    for gparam, param, gsum in zip(gparams, params, gsums):
        updates[gsum] =  T.cast(gsum + (gparam ** as_floatX(2.)), "float32")
        updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), "float32")

    return updates, lr

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)

    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param
    return updates
