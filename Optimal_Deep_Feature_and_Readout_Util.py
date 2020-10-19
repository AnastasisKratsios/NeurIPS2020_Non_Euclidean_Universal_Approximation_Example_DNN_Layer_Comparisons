#!/usr/bin/env python
# coding: utf-8

# ## Helper Function(s):

# In[1]:


# Univariate Compositer
def compositer_univariate(x):
    # Apply Activation
    x_out = 1 / (1 + np.math.exp(-x))
    # Output
    return x_out

# Vectorize
compositer = np.vectorize(compositer_univariate)

def activ_univariate(x):
    if x<0:
        x_out = 0.9*np.sign(x)*np.math.pow(-x,2)
    else:
        x_out = 0.1*np.math.pow(x,2)
    return x_out

# Vectorize
activ = np.vectorize(activ_univariate)

def activ_univariate_inv(x):
    if x<0:
        x_out = (1/0.9)*np.sign(x)*np.sqrt(-x)
    else:
        x_out = (1/0.1)*np.sqrt(x)
    return x_out

# Vectorize
activ_inv = np.vectorize(activ_univariate_inv)


# ## Custom Layers
#  - Fully Conneted Dense: Typical Feed-Forward Layer
#  - Fully Connected Dense Invertible: Necessarily satisfies for input and output layer(s)
#  - Fully Connected Dense Destructor: Violates Assumptions for both input and ouput layer(s) (it is neither injective nor surjective)

# In[1]:


class fullyConnected_Dense(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
class fullyConnected_Dense_Invertible(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                               initializer='zeros',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='zeros',
                               trainable=True)

    def call(self, inputs):
        expw = tf.linalg.expm(self.w)
        return tf.matmul(inputs, expw) + self.b
    
class fullyConnected_Dense_Desctructor(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Desctructor, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='zeros',
                               trainable=True)

    def call(self, inputs):
        badw = tf.matmul(self.w,self.w)
        return tf.matmul(inputs, badw) + self.b


# In[ ]:


class fullyConnected_Dense_Random(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Random, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=False)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=False)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

