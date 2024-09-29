import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Attention(layers.Layer):
    def __init__(self, d_model, d_k, d_v, **kwargs):
        super(Attention, self).__init__()
        self.fc_q = layers.Dense(d_k)
        self.fc_k = layers.Dense(d_k)
        self.fc_v = layers.Dense(d_k)
        self.fc_o = layers.Dense(d_model)
        self.softmax = tf.nn.softmax

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v


    def call(self, queries, keys, values, attention_mask=None, attention_weights=None):

        q = self.fc_q(queries)  
        k = self.fc_k(keys)    
        v = self.fc_v(values) 

        att = tf.matmul(q, k, transpose_b=True) / np.sqrt(self.d_k) 

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = tf.multiply(att, attention_mask)

        att = self.softmax(att, -1) 

        out = tf.matmul(att, v)  
        return self.fc_o(out)
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_v
        })
        return config


if __name__ == '__main__':
    # input = tf.random.normal((50, 49, 512))
    # sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    input = tf.random.normal((10, 21))
    sa = Attention(d_model=21, d_k=21, d_v=21)
    output = sa(input, input, input)
    print(output.shape)