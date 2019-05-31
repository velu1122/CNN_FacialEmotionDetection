"""
@author: velmurugan.jeyaram
"""
# CNN based Emotion Detection
    

import numpy as np
import tensorflow as tf
import math
from dataset import EmotionData


def randomize(self):
        c = np.c_[self.training_images.reshape(len(self.training_images), -1), self.training_labels.reshape(len(self.training_labels), -1)]
        np.random.shuffle(c)
        a2 = c[:, :self.training_images.size//len(self.training_images)].reshape(self.training_images.shape)
        b2 = c[:, self.training_images.size//len(self.training_images):].reshape(self.training_labels.shape)
        
        self.training_images = a2
        self.training_labels = b2
        
'''
#Alternate way for randomization 
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]
'''        

# Creating the Model
class CNNLayer():
        
    def __init__(self):
        # Create 2 placeholders, x and y_true.
        # x shape = [None,32,32,3]
        # y_true shape = [None,10]
        self.x = tf.placeholder(tf.float32,shape=[None,32,32,3], name='x')
        self.y_true = tf.placeholder(tf.float32,shape=[None,5],name='y_true')        
                
        '''Create one more placeholder called hold_prob. No need for shape here. 
        This placeholder will just hold a single probability for the dropout.'''
        self.hold_prob = tf.placeholder(tf.float32,name='hold_prob')

    
    # Parameters initialization 
    # * init_weights
    # * init_bias
    # * conv2d
    # * max_pool_2by2
    # * convolutional_layer
    # * normal_full_layer
    def init_weights(self, shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)
    
    def init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2by2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    
    def convolutional_layer(self, input_x, shape):
        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])
        return tf.nn.relu(self.conv2d(input_x, W) + b)
    
    def normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.matmul(input_layer, W) + b


    def create_graph(self):
        #Create the Layers
        # Create a convolutional layer and a pooling layer
        convo_1 = self.convolutional_layer(self.x,shape=[4,4,3,32])
        convo_1_pooling = self.max_pool_2by2(convo_1)
                
        #Create the next convolutional and pooling layers.  The last two dimensions of the convo_2 layer should be 32,64
        convo_2 = self.convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
        convo_2_pooling = self.max_pool_2by2(convo_2)      
        
        #Now create a flattened layer by reshaping the pooling layer into [-1,8 \* 8 \* 64] or [-1,4096]
        convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64]) 
        
        #Create a new full layer using the normal_full_layer function and passing in your flattend convolutional 2 layer with size=1024.
        full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat,1024))
        
        # Now create the dropout layer with tf.nn.dropout, remember to pass in your hold_prob placeholder.
        full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=self.hold_prob)     
        
        #Set the output to y_pred by passing in the dropout layer into the normal_full_layer function. The size should be 10 because of the 10 possible labels
        self.y_pred = self.normal_full_layer(full_one_dropout,5)
       
        #Loss Function
        #Create a cross_entropy loss function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,logits=self.y_pred))
                
        # Optimizer
        # Create the optimizer using an Adam Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        return train

# Create a variable to intialize all the global tf variables
    def start_session(self):
        # Initialze the datasets
        ch = EmotionData()
        ch.make_sets()
        
        # Graph Session
        # Perform the training and test printouts
        train = self.create_graph()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batchsz = 100
            epochs = 50
            
            for epoch in range(epochs):
                randomize(ch)
                ch.i = 0
                for i in range(math.ceil(len(ch.training_labels)/batchsz)):
                    batch = ch.next_batch(batchsz)
                    sess.run(train, feed_dict={self.x: batch[0], self.y_true: batch[1], self.hold_prob: 0.5})
        
                    # PRINT OUT A MESSAGE EVERY 100 STEPS
                    if i%50 == 0:
        
                        print('Currently on step {}:{}'.format(epoch,i))
                        print('Accuracy is:')
                        # Test the Train Model
                        matches = tf.equal(tf.argmax(self.y_pred,1),tf.argmax(self.y_true,1))
                        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
                        print(sess.run(acc,feed_dict={self.x:ch.test_images,self.y_true:ch.test_labels,self.hold_prob:1.0}))
                        print('\n')
            
            # Save the variables to disk.
            saver = tf.train.Saver()
            save_path = saver.save(sess, "./Savedmodel/Emotiondetection.ckpt")
            print("Model saved in path: %s" % save_path)

            # Saving the model files (Graph + Variables/parameters)
            builder = tf.saved_model.builder.SavedModelBuilder("./Savedmodel/model/")
            builder.add_meta_graph_and_variables(sess, ["Emotions"], signature_def_map= {
            "model": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs= {"x": self.x}, outputs= {"finalnode": self.y_pred}) })
            builder.save()

if __name__ == "__main__":
    instance = CNNLayer()
    instance.start_session()
