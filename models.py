import tensorflow as tf

def F1_loss(labels,pred,from_logits=True,epsilon=1e-7):
    labels = tf.cast(labels,tf.float32)
    if from_logits:
        pred = tf.nn.softmax(pred,axis=-1)
    truepos = tf.reduce_sum(labels * pred,axis=0)
    falsepos = tf.reduce_sum((1-labels) * pred,axis=0)
    falseneg = tf.reduce_sum(labels * (1-pred),axis=0)
    precision = truepos / (truepos + falsepos + epsilon)
    recall = truepos / (truepos + falseneg + epsilon)
    F1 = 2 * precision * recall / (precision + recall + epsilon)
    F1 = tf.where(tf.math.is_nan(F1),tf.zeros_like(F1),F1)
    return 1 - tf.reduce_mean(F1)

class SE_Block(tf.keras.layers.Layer):
    def __init__(self,out_channels,ratio):
        super(SE_Block,self).__init__()
        assert not out_channels % ratio,"SE_Block: ratio should divide out_channels"
        self.S = tf.keras.layers.GlobalAveragePooling2D()
        self.E1 = tf.keras.layers.Dense(out_channels // ratio,activation='relu')
        self.E2 = tf.keras.layers.Dense(out_channels,activation='sigmoid')
        self.R = tf.keras.layers.Reshape((1,1,out_channels))
        pass
    
    @tf.function
    def call(self,inputs):
        squeeze = self.S(inputs)
        excitation = self.R(self.E2(self.E1(squeeze)))
        scale = inputs * excitation
        return scale

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,bn_channels,out_channels):
        super(Bottleneck,self).__init__()
        
        self.C1 = tf.keras.layers.Conv2D(bn_channels,1,padding='SAME',use_bias=False)
        self.B1 = tf.keras.layers.BatchNormalization()
        self.A1 = tf.keras.layers.Activation('relu')

        self.DC = tf.keras.layers.DepthwiseConv2D(2,padding='SAME',use_bias=False)
        self.B2 = tf.keras.layers.BatchNormalization()
        self.A2 = tf.keras.layers.Activation('relu')

        self.C3 = tf.keras.layers.Conv2D(out_channels,1,padding='SAME',use_bias=False)
        self.B3 = tf.keras.layers.BatchNormalization()
        pass

    @tf.function
    def call(self,inputs,training):
        conv1 = self.A1(self.B1(self.C1(inputs),training))
        grouped = self.A2(self.B2(self.DC(conv1),training))
        conv2 = self.B3(self.C3(grouped),training)
        return conv2

class ResNeXt_Block(tf.keras.layers.Layer):
    def __init__(self,cardinality,in_channels,out_channels,expansion):
        super(ResNeXt_Block,self).__init__()

        self.in_channels = in_channels
        self.conv_input = in_channels != out_channels

        assert not out_channels % (expansion * cardinality),"(expansion * cardinality) must divide out_channels"
        
        if self.conv_input:
            self.C0 = tf.keras.layers.Conv2D(out_channels,1,padding='SAME',use_bias=False)
            self.B0 = tf.keras.layers.BatchNormalization()
            self.A0 = tf.keras.layers.Activation('relu')

        self.BN = []
        for i in range(cardinality):
            self.BN.append(Bottleneck((out_channels // cardinality) // expansion,out_channels))

        self.A1 = tf.keras.layers.Activation('relu')
        pass
    
    @tf.function
    def se_call(self,inputs,training):
        x = inputs
        groups = [bn.call(x,training) for bn in self.BN]
        grouped = tf.reduce_sum(groups,axis=0)
        if self.conv_input:
            x = self.A0(self.B0(self.C0(x),training))
        return x,grouped

    @tf.function
    def call(self,inputs,training):
        x,grouped = self.se_call(inputs,training)
        return self.A1(x + grouped)

class SE_ResNeXt_Block(tf.keras.layers.Layer):
    def __init__(self,cardinality,in_channels,out_channels,expansion,ratio):
        super(SE_ResNeXt_Block,self).__init__()
        
        self.RB = ResNeXt_Block(cardinality,in_channels,out_channels,expansion)
        self.SE = SE_Block(out_channels,ratio)
        pass
    
    @tf.function
    def call(self,inputs,training):
        x,grouped = self.RB.se_call(inputs,training)
        seblock = self.SE.call(grouped)
        return self.RB.A1(x + seblock)

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
        return F1_loss(labels,logits)

    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))

class SENet(tf.keras.Model):
    def __init__(self):
        super(SENet,self).__init__()

        self.num_classes = 2
        
        self.C1 = tf.keras.layers.Conv2D(16,5,padding='SAME',strides=2,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.SE1 = SE_Block(16,4)
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.C2 = tf.keras.layers.Conv2D(20,5,padding='SAME',strides=1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.SE2 = SE_Block(20,4)
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')

        self.C3 = tf.keras.layers.Conv2D(20,5,padding='SAME',strides=1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.B3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.SE3 = SE_Block(20,4)
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
        return F1_loss(labels,logits)
    
    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet,self).__init__()

        self.num_input_channels = 3
        self.num_classes = 2
        
        self.R1 = ResNeXt_Block(1,self.num_input_channels,16,4)
        self.B1 = tf.keras.layers.BatchNormalization()
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.R2 = ResNeXt_Block(1,16,20,4)
        self.B2 = tf.keras.layers.BatchNormalization()
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')
        '''
        self.R3 = ResNeXt_Block(1,20,20,4)
        self.B3 = tf.keras.layers.BatchNormalization()
        '''
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
        block1 = self.P1(self.B1(self.R1.call(inputs,training),training))
        block2 = self.P2(self.B2(self.R2.call(block1,training),training))
        #block3 = self.flatten(self.B3(self.R3.call(block2,training),training))
        dense1 = self.D1(self.F1(self.flatten(block2)),training)
        dense2 = self.D2(self.F2(dense1),training)
        output = self.F3(dense2)
        return output

    def loss(self,logits,labels):
        return F1_loss(labels,logits)
    
    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))
    
class SE_ResNet(tf.keras.Model):
    def __init__(self):
        super(SE_ResNet,self).__init__()

        self.num_input_channels = 3
        self.num_classes = 2
        
        self.R1 = SE_ResNeXt_Block(1,self.num_input_channels,16,4,4)
        self.B1 = tf.keras.layers.BatchNormalization()
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.R2 = SE_ResNeXt_Block(1,16,20,4,4)
        self.B2 = tf.keras.layers.BatchNormalization()
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')
        '''
        self.R3 = SE_ResNeXt_Block(1,20,20,4,4)
        self.B3 = tf.keras.layers.BatchNormalization()
        '''
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
        block1 = self.P1(self.B1(self.R1.call(inputs,training),training))
        block2 = self.P2(self.B2(self.R2.call(block1,training),training))
        #block3 = self.flatten(self.B3(self.R3.call(block2,training),training))
        dense1 = self.D1(self.F1(self.flatten(block2)),training)
        dense2 = self.D2(self.F2(dense1),training)
        output = self.F3(dense2)
        return output

    def loss(self,logits,labels):
        return F1_loss(labels,logits)
    
    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))

class ResNeXt(tf.keras.Model):
    def __init__(self):
        super(ResNeXt,self).__init__()
        
        self.num_input_channels = 3
        self.num_classes = 2
        
        self.R1 = ResNeXt_Block(4,self.num_input_channels,16,2)
        self.B1 = tf.keras.layers.BatchNormalization()
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.R2 = ResNeXt_Block(5,16,20,2)
        self.B2 = tf.keras.layers.BatchNormalization()
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')
        '''
        self.R3 = ResNeXt_Block(5,20,20,2)
        self.B3 = tf.keras.layers.BatchNormalization()
        '''
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
        block1 = self.P1(self.B1(self.R1.call(inputs,training),training))
        block2 = self.P2(self.B2(self.R2.call(block1,training),training))
        #block3 = self.flatten(self.B3(self.R3.call(block2,training),training))
        dense1 = self.D1(self.F1(self.flatten(block2)),training)
        dense2 = self.D2(self.F2(dense1),training)
        output = self.F3(dense2)
        return output

    def loss(self,logits,labels):
        return F1_loss(labels,logits)
    
    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))

class SE_ResNeXt(tf.keras.Model):
    def __init__(self):
        super(SE_ResNeXt,self).__init__()

        self.num_input_channels = 3
        self.num_classes = 2
        
        self.R1 = SE_ResNeXt_Block(4,self.num_input_channels,16,2,4)
        self.B1 = tf.keras.layers.BatchNormalization()
        self.P1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='VALID')

        self.R2 = SE_ResNeXt_Block(5,16,20,2,4)
        self.B2 = tf.keras.layers.BatchNormalization()
        self.P2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='VALID')
        '''
        self.R3 = SE_ResNeXt_Block(5,20,20,2,4)
        self.B3 = tf.keras.layers.BatchNormalization()
        '''
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
        block1 = self.P1(self.B1(self.R1.call(inputs,training),training))
        block2 = self.P2(self.B2(self.R2.call(block1,training),training))
        #block3 = self.flatten(self.B3(self.R3.call(block2,training),training))
        dense1 = self.D1(self.F1(self.flatten(block2)),training)
        dense2 = self.D2(self.F2(dense1),training)
        output = self.F3(dense2)
        return output

    def loss(self,logits,labels):
        return F1_loss(labels,logits)
    
    def accuracy(self,logits,labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)), dtype=tf.float32))
