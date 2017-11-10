import tensorflow as tf

#with tf.Session() as sess:
#    x = tf.placeholder(tf.float32, [None, 32], name="x")
#    y = tf.placeholder(tf.float32, [None, 8], name="y")

#    w1 = tf.Variable(tf.truncated_normal([32, 16], stddev=0.1))
#    b1 = tf.Variable(tf.constant(0.0, shape=[16]))

#    w2 = tf.Variable(tf.truncated_normal([16, 8], stddev=0.1))
#    b2 = tf.Variable(tf.constant(0.0, shape=[8]))

#    a = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, w1), b1))
#    y_out = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a, w2), b2), name="y_out")
#    cost = tf.reduce_sum(tf.square(y-y_out), name="cost")
#    optimizer = tf.train.AdamOptimizer().minimize(cost, name="train")

#    init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')
#    tf.train.write_graph(sess.graph_def, './', 'mlp.pb', as_text=False)


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 3], name="x")
    y = tf.placeholder(tf.float32, [None, 4], name="y")

    w1 = tf.Variable(tf.truncated_normal([3, 16], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.0, shape=[16]))

    w2 = tf.Variable(tf.truncated_normal([16, 32], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.0, shape=[32]))

    w3 = tf.Variable(tf.truncated_normal([32, 4], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.0, shape=[4]))

    a = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w1), b1))
    a = tf.nn.relu(tf.nn.bias_add(tf.matmul(a, w2), b2))
    y_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(a, w3), b3), name="y_out")
    y_argout = tf.argmax(input=y_out, axis=1, name="y_argout")
    logits = y_out
    #cost = tf.reduce_sum(tf.square(y-y_out), name="cost")
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss, name="train")

    # init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')
    init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
    tf.train.write_graph(sess.graph_def, './', 'mlp2.pb', as_text=False)
