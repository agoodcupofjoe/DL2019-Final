import tensorflow as tf

class SE_Block(tf.keras.layers.Layer):
    def __init__(self,in_channels,out_channels,ratio):
        super(SE_Block,self).__init__()
        self.S = tf.keras.layers.GlobalAveragePooling2D()
        self.E1 = tf.keras.layers.Dense(in_channels // ratio,activation='relu')
        self.E2 = tf.keras.layers.Dense(out_channels,activation='sigmoid')
        self.R = tf.keras.layers.Reshape((1,1,out_channels))
        self.M = tf.keras.layers.Multiply()
        pass
    
    @tf.function
    def call(self,inputs,training=True):
        squeeze = self.S(inputs)
        excitation = self.R(self.E2(self.E1(squeeze)))
        scale = self.M([inputs,excitation])
        return scale

class SE_ResNeXt_Block(tf.keras.layers.Layer):
    def __init__(self,in_channels,out_channels,ratio,cardinality,strides=1):#might need more params
        super(SE_ResNeXt_Block,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.cardinality = cardinality
        self.strides = strides
        
        n = out_channels // cardinality
        assert not n, "Cardinality error"
        self.G = []
        self.C = []
        for i in range(cardinality):
            group = tf.keras.layers.Lambda(lambda x: x[:,:,:,i*n:(i+1)*n])
            conv = tf.keras.layers.Conv2D(n,kernel_size=3,strides=strides,padding='SAME',use_bias=False)
            self.G.append(group)
            self.C.append(conv)

        self.C1 = tf.keras.layers.Conv2D(in_channels,kernel_size=1,strides=1,padding='SAME',use_bias=False)
        self.B1 = tf.keras.layers.BatchNormalization()
        if strides != 1:
            self.C2 = tf.keras.layers.Conv2D(out_channels,kernel_size=1,strides=strides,padding='SAME',use_bias=False)
            self.B2 = tf.keras.layers.BatchNormalization()

        self.A = tf.keras.layers.Add()
        self.R = tf.keras.layers.LeakyReLU()
        
        self.SE = SE_Block(in_channels,out_channels,ratio)
        pass
    
    @tf.function
    def call(self,inputs,training=True):
        x = inputs
        conv1 = self.B1(self.C1(x),training)
        groups = []
        for i in range(self.cardinality):
            group = self.C[i](self.G[i](conv1))
            groups.append(group)
        grouped = tf.keras.layers.concatenate(groups) #set axis? default axis=-1
        seblock = self.SE.call(grouped,training=training)
        if self.strides != 1:
            x = self.B2(self.C2(x),training)
        return self.R(self.A([x,seblock]))

class CNN(tf.keras.Model):
    def __init__(self):

        super(CNN,self).__init__()

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

class SENet(tf.keras.Model):
    def __init__(self):
        super(SENet,self).__init__()

        self.num_classes = 2
        
        self.C1 = tf.keras.layers.Conv2D(16,5,padding='SAME',strides=2,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.SE1 = SE_Block(16,16,4)
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.C2 = tf.keras.layers.Conv2D(20,5,padding='SAME',strides=1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.SE2 = SE_Block(20,20,4)
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')

        self.C3 = tf.keras.layers.Conv2D(20,5,padding='SAME',strides=1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.SE3 = SE_Block(20,20,4)
        self.flatten = tf.keras.layers.Flatten()

        self.F1 = tf.keras.layers.Dense(320,activation='relu',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.D1 = tf.keras.layers.Dropout(0.3)

        self.F2 = tf.keras.layers.Dense(160,activation='relu',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.D2 = tf.keras.layers.Dropout(0.3)

        self.F3 = tf.keras.layers.Dense(self.num_classes,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        pass

    @tf.function
    def call(self,inputs,training):
        # First convolution/maxpooling block
        conv1 = self.P1(self.SE1(self.B1(tf.nn.relu(self.C1(inputs)),training)))

        # Second convolution/maxpooling block
        conv2 = self.P2(self.SE2(self.B2(tf.nn.relu(self.C2(conv1)),training)))

        # Third convolution/maxpooling block
        conv3 = self.flatten(self.SE3(self.B3(tf.nn.relu(self.C3(conv2)),training)))

        # Fully connected layers with dropout
        dense1 = self.D1(self.F1(conv3),training)
        dense2 = self.D2(self.F2(dense1),training)
        dense3 = self.F3(dense2)

        # Return output of final layer as logits
        return dense3

    def loss(self,logits,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))
    
    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))
