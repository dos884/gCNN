import time

import numpy as np
import tensorflow as tf

IM_SIZE = 28


class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters1 = []
        self.filters2 = []
        for i in range(6):
            m = tf.keras.Sequential([tf.keras.layers.Dense(units=12, activation='tanh'),
                                     tf.keras.layers.Dense(units=6, activation='tanh'),
                                     tf.keras.layers.Dense(units=1, activation='tanh')])
            self.filters1.append(m)
        for i in range(16):
            m = tf.keras.Sequential([tf.keras.layers.Dense(units=12, activation='tanh'),
                                     tf.keras.layers.Dense(units=6, activation='tanh'),
                                     tf.keras.layers.Dense(units=1, activation='tanh')])
            self.filters2.append(m)
        self.final_layers = tf.keras.Sequential([tf.keras.layers.Dense(120, activation='tanh'),
                                                 tf.keras.layers.Dense(84, activation='tanh'),
                                                 tf.keras.layers.Dense(10, activation='softmax')])

    # @tf.function
    def call(self, input, training=None, mask=None):

        patches = tf.image.extract_patches(input, sizes=[1, 5, 5, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                          padding='SAME')
        reshaped = tf.reshape(patches, [-1, patches.shape[1]*patches.shape[2], patches.shape[-1]])
        preds1 = []
        size_after_pooling=0
        for i in range(6):
            after_filter = self.filters1[i](reshaped)
            resh = tf.reshape(after_filter,[1,IM_SIZE,IM_SIZE,1])
            avg_pooled = tf.keras.layers.AveragePooling2D()(resh)
            size_after_pooling = avg_pooled.shape[1]
            patched = tf.image.extract_patches(avg_pooled,sizes=[1,5,5,1],strides=[1,1,1,1],rates=[1,1,1,1],padding="SAME")
            reshed = tf.reshape(patched, [-1, patched.shape[1]*patched.shape[2], patched.shape[-1]])
            preds1.append(reshed)
        preds2 = []
        for i in range(16):
            for j in range(6):
                after_filter2 = self.filters2[i](preds1[j])
                resh2= tf.reshape(after_filter2,[1,size_after_pooling,size_after_pooling,1])
                avg_pooled2=tf.keras.layers.AveragePooling2D()(resh2)
                flattened = tf.keras.layers.Flatten()(avg_pooled2)
                preds2.append(flattened)
        concated = tf.concat(values=preds2, axis=1)
        final = self.final_layers(concated)
        return final

    # @property
    def trainable_variables(self):
        vars = []
        for i in range(6):
            vars.extend(self.filters1[i].trainable_weights)
        for i in range(16):
            vars.extend(self.filters2[i].trainable_weights)
        vars.extend(self.final_layers.trainable_weights)
        return vars

    def set_weights(self, weights):
        counter = 0
        for i in range(6):
            self.filters1[i].set_weights(weights[counter*6:(counter+1)*6])
            counter += 1
        for i in range(16):
            self.filters2[i].set_weights(weights[counter*6:(counter+1)*6])
            counter += 1
        self.final_layers.set_weights(weights[counter*6:(counter+1)*6])
        # print(counter)

    def batches_loss(self, xs, ys):
        loss = 0
        for i in range(len(xs)):
            x=xs[i]
            y=ys[i]
            pred = self.call(x)
            loss += tf.keras.losses.sparse_categorical_crossentropy(y, pred, from_logits=True)
        return loss


# def build_dataset(batch_size):
#     single_img = build_test_img()
#     dataset = tf.data.Dataset.from_tensors(single_img)
#     dataset = dataset.repeat()
#     dataset = dataset.batch(batch_size)
#     return dataset
#
#
# def build_test_img():
#     single_img = tf.reshape(tf.range(0, 25), [5, 5, 1])
#     single_img = tf.cast(single_img, dtype=tf.float32)
#     return single_img


def main():
    model = MyModel()
    x = tf.reshape(tf.range(1, IM_SIZE**2+1), [1, IM_SIZE, IM_SIZE, 1])
    x_batch = [x,x,x]
    ys=[1,1,1]
    # test = tf.reshape(tf.range(0,9),[3,3])
    output = model.call(x)
    losses = []
    learning_rate = 1e-3

    for step in range(100):
        # total1 = time.time()
        with tf.GradientTape() as tape:
            # t1 = time.time()
            loss_value = model.batches_loss(x_batch, ys)
            # t2 = time.time()
        # print(f"loss calculation time: {(t2-t1)}")
        losses.append(loss_value)
        V = model.trainable_variables()
        grads = tape.gradient(loss_value, model.trainable_variables())
        for i in range(len(V)):
            V[i]=V[i]-learning_rate*grads[i]
        model.set_weights(V)
        print(loss_value)
        # total2 = time.time()
        # print(f"total iteration time is: {(total2-total1)}")
    # model.final_layers.set_weights(W)
    # print(output)
    print("bye")


if __name__ == '__main__':
    main()

# test_conv=tf.keras.layers.Conv2D(filters=1,kernel_size=(1,1))
# output=test_conv(patches)
# print(output)
# print(output.shape)
# print(test_conv.kernel.shape)

