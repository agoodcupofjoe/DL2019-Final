import tensorflow as tf

class Baseline(tf.keras.Model):
    def __init__(self):

        super(Baseline,self).__init__()

        self.batch_size = 500

        self.num_classes = 2
        self.num_epochs = 15
        self.image_height = 300
        self.image_width = 400

        self.C1 = tf.keras.layers.Conv2D(16,5,padding='SAME',strides=2,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.C2 = tf.keras.layers.Conv2D(20,5,padding='SAME',strides=1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')

        self.C3 = tf.keras.layers.Conv2D(20,5,padding='SAME',strides=1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.flatten = tf.keras.layers.Flatten()

        self.F1 = tf.keras.layers.Dense(320,activation='relu',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.D1 = tf.keras.layers.Dropout(0.3)

        self.F2 = tf.keras.layers.Dense(160,activation='relu',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.D2 = tf.keras.layers.Dropout(0.3)

        self.F3 = tf.keras.layers.Dense(self.num_classes,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def call(self,inputs,training):
        # First convolution layer
        conv1 = self.C1(inputs)
        relu1 = tf.nn.relu(conv1)
        batchnorm1 = self.B1(relu1,training)
        maxpool1 = self.P1(batchnorm1)

        # Second convolution layer
        conv2 = self.C2(maxpool1)
        relu2 = tf.nn.relu(conv2)
        batchnorm2 = self.B2(relu2,training)
        maxpool2 = self.P2(batchnorm2)

        # Third convolution layer
        conv3 = self.C3(maxpool2)
        relu3 = tf.nn.relu(conv3)
        batchnorm3 = self.B3(relu3,training)

        # Sequence of dense layers to produce label prediction
        dense1 = self.D1(self.F1(self.flatten(batchnorm3)),training)
        dense2 = self.D2(self.F2(dense1),training)
        dense3 = self.F3(dense2)

        # Return the output of the final dense layer
        return dense3

    def loss(self,logits,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))

    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))