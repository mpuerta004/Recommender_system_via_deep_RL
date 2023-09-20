import tensorflow as tf
import numpy as np

class DRRAveStateRepresentation(tf.keras.Model):
    # embedding_dim = la dimensionalidad del vector de entrada 
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        #para concatenar
        self.concat = tf.keras.layers.Concatenate()
        #para aplanar el vector
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        # X = [user_caracteristics (item), h_t=[items with a positive past iterations]]
        # To take the transpose of the matrices in dimension-0 (such as when you are transposing matrices 
        # where 0 is the batch dimension), you would set perm=[0,2,1].
        # https://www.tensorflow.org/api_docs/python/tf/transpose
        items_eb = tf.transpose(x[1], perm=(0,2,1))/self.embedding_dim
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0,2,1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([x[0], wav])
        concat = self.concat([x[0], user_wav, wav])
        #Dimensionalidad de salida 3*embedding_dim
        return self.flatten(concat)