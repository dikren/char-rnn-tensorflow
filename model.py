import tensorflow as tf
# from tensorflow.models.rnn import rnn_cell
# from tensorflow.nn.rnn_cell
# import tensorflow.nn.rnn_cell as rnn_cell
# from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.nn.seq2seq

import numpy as np

from tensorflow.python.ops import variable_scope as vs



def _linear_with_normalization(args, output_size, bias, bias_start=0.0, scope=None):
  """
  Normolized value is computed bu formua norm_val = beta + gamma * (tensor * W - mean) / sqrt(variance + epsilon)
  without normalization: tensor * W
  all args then comvined: sum_i(val[i]) + b
  For each tensor will be computed separate normalization.

  Args:
    args: dict or list of dicts with nesxt fields:
        tensor: a 2D Tensor of size: batch x n
        normolized:
            "No" - do not use batch norm on this tensor
            "WithFixGamma" - use batch norm with specified gamma
            "WithLearningGamma" - use batch. Parameter gamma is learned by net itself. gamma specified here (as function argument) stores initial value of network's parameter
        gamma: if needed, gamma parameter cpecified here
        epsilon: if needed, epsilon must be specified here
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (rnn_cell._is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not rnn_cell._is_sequence(args):
    args = [args]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    results = 0
    for i in range(0, len(args)):
        tensor = args[i]["tensor"]
        normolized = args[i]["normolized"]

        matrix = vs.get_variable("Matrix_" + str(i), [tensor.get_shape().as_list()[1], output_size])
        res = math_ops.matmul(tensor, matrix)

        if normolized == "No":
            result = res
        else:
            epsilon = args[i]["epsilon"]
            gamma = args[i]["gamma"]

            if normolized == "WithFixGamma":
                pass
            elif normolized == "WithLearningGamma":
                gamma = vs.get_variable("Gamma_" + str(i), [output_size], initializer=init_ops.constant_initializer(gamma))
            else:
                raise Exception("Unsupported parameter value: " + str(normolized))

            mean = math_ops.reduce_mean(res, 0, keep_dims=True)
            var = math_ops.reduce_mean(math_ops.square(res - mean), 0, keep_dims=True) + epsilon
            std = math_ops.sqrt(var)
            result = gamma * (res - mean) / std

        results += result

    if not bias:
      return results
    bias_term = vs.get_variable(
        "Bias", [output_size],
        initializer=init_ops.constant_initializer(bias_start))
  return results + bias_term




def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """
  Normolized value is computed bu formua norm_val = beta + gamma * (val - mean) / sqrt(variance + epsilon)
  Linear map with normalization: norm(sum_i(args[i] * W[i])), where W[i] is a variable.
  For each tensor will be computed separate normalization.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    gammas: vector of gamma parameters.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (rnn_cell._is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not rnn_cell._is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    results = 0
    for i in range(0, len(args)):

        matrix = vs.get_variable("Matrix_" + str(i), [args[i].get_shape().as_list()[1], output_size])
        res = math_ops.matmul(args[i], matrix)

        results += res

    if not bias:
      return results
    bias_term = vs.get_variable(
        "Bias", [output_size],
        initializer=init_ops.constant_initializer(bias_start))
  return results + bias_term




class ResNetBasicRNNCell(rnn_cell.BasicRNNCell):
    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            RNN_output = self._activation(_linear_with_normalization(
                [
                 {"tensor": inputs,
                  "normolized": "WithFixGamma",
                  "gamma": 1,
                  "epsilon": 0.001},

                    # {"tensor": inputs,
                    #  "normolized": "No"},




                 # {"tensor": state,
                 #  "normolized": "WithFixGamma",
                 #  "gamma": 0.1,
                 #  "epsilon": 0.001}

                 # {"tensor": state,
                 #  "normolized": "No"}

                 {"tensor": state,
                  "normolized": "WithLearningGamma",
                  "gamma": 0.1,
                  "epsilon": 0.001}
                 ],
                output_size=self._num_units, bias=True))
            # RNN_output = self._activation(_linear([inputs, state],   output_size=self._num_units, bias=True))
            output = RNN_output # + state
        return output, output

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            # cell_fn = rnn_cell.BasicRNNCell
            cell_fn = ResNetBasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret


