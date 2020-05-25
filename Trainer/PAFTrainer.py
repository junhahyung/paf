import os
import io
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import trange
from utils.losses import *

"""
PAF Trainer
"""
class Trainer():
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.config = config
        self.callbacks = list()

        # init model parts
        self.paf_gen = self.model.paf_gen
        # beta 10??
        #self.paf_gen_loss = weighted_cross_entropy(beta=10)
        self.paf_gen_loss = dist
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.000005, 4 * self.config.training_steps, 0.90, staircase=True)
        self.paf_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        #init dataset
        self.dataset = iter(train_data())
        self.val_dataset = iter(val_data())

        # init tensorboard
        self.tb_writer = tf.summary.create_file_writer(self.config.tensorboard_log_dir, name='train')
        self.val_tb_writer = tf.summary.create_file_writer(self.config.tensorboard_log_dir, name='val')

        self.imgs_from_train = None
        self.imgs_from_val = None


    '''
    def load_weight(self):
        model_list = os.listdir(os.path.join('experiments/', self.config.pretrained_exp, 'checkpoints'))
        model_list.sort()

        generator_model_path = model_list[-2]
        self.model.generator.load_weights(os.path.join('experiments/', self.config.pretrained_exp, 'checkpoints', generator_model_path))
        print(f"Loaded pretrained {self.config.pretrained_exp} generator weights.")
        '''

    def get_loss(self, model, x, y, loss_function, training=False):
        y_ = model(x, training=training)
        return loss_function(y_true=y, y_pred=y_), y_

    @tf.function()
    def train_step(self, model, x, y, loss_function):
        with tf.GradientTape() as tape:
            loss_value, y_ = self.get_loss(model, x, y, loss_function, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), y_


    def train_epoch(self, epoch):
        t = trange(self.config.training_steps)
        losses = []

        for i in t:
            x, y = next(self.dataset)

            if (i == 0) & (epoch == 0):
                self.imgs_from_train = x

            # train generator
            loss, grads, paf = self.train_step(self.paf_gen, x, y[1], self.paf_gen_loss)
            lnan = tf.math.is_nan(loss)
            lnan = tf.math.reduce_any(lnan)
            if lnan:
                print("loss is nan")
                xnan = tf.math.is_nan(x)
                xnan = tf.math.reduce_any(xnan)
                ynan = tf.math.is_nan(y[1])
                ynan = tf.math.reduce_any(ynan)
                pnan = tf.math.is_nan(paf)
                pnan = tf.math.reduce_any(pnan)
                if xnan:
                    print("x has nan")
                else:
                    print("x is NOT nan")
                if ynan:
                    print("y has nan")
                else:
                    print("y is NOT nan")
                if pnan:
                    print("p has nan")
                else:
                    print("p is NOT nan")

            self.paf_opt.apply_gradients(zip(grads, self.paf_gen.trainable_variables))

            losses.append(loss)

        training_loss = sum(losses) / len(losses)

        val_t = trange(self.config.validation_steps)
        val_losses = []
        for i in val_t:
            x, y = next(self.val_dataset)
            if (i == 0) & (epoch == 0):
                self.imgs_from_val = x

            loss, paf_ = self.get_loss(self.paf_gen, x, y[1], self.paf_gen_loss)

            val_losses.append(loss)
        val_loss = sum(val_losses) / len(val_losses)

        print(losses)

        print(f"Epoch {epoch:03d}: train-loss: {training_loss:.06f} "
              f"--- val-loss: {val_loss:.06f}" )

        # tf.summary
        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def plot_quiver(img, paf):
            figure = plt.figure(figsize=(10,10))
            paf = tf.transpose(paf, (2,0,1))
            xi = np.arange(240)
            yi = np.arange(240)
            plt.imshow(img)
            plt.quiver(xi, yi, np.transpose(paf[0]), -np.transpose(paf[1]), scale_units='xy', scale=0.1)

            return figure

        with self.tb_writer.as_default():
            # scalar
            tf.summary.scalar('MSE', training_loss, step=epoch)
            tf.summary.scalar('Val MSE', val_loss, step=epoch)

            # record images
            i = 0
            x = self.imgs_from_train
            paf = self.paf_gen(self.imgs_from_train)

            input_img = x[i]
            paf = paf[i]
            
            figure = plot_quiver(input_img, paf)
            input_img = tf.reshape(input_img, (1, 240, 240, 3))
            '''
            paf = tf.reshape(paf, (1, 240, 240, 2))

            tf_img = tf.concat([input_img, paf], axis=-1)
            '''
            tf.summary.image('paf', plot_to_image(figure), step=epoch)
            tf.summary.image('Train', input_img, step=epoch)

            x = self.imgs_from_val
            paf = self.paf_gen(self.imgs_from_val)
            
            input_img = x[i]
            paf = paf[i]

            figure = plot_quiver(input_img, paf)
            input_img = tf.reshape(input_img, (1, 240, 240, 3))
            '''
            paf = tf.reshape(paf, (1, 240,240, 2))

            tf_img = tf.concat([input_img, paf], axis=-1)
            '''
            tf.summary.image('paf val', plot_to_image(figure), step=epoch)
            tf.summary.image('Validation', input_img, step=epoch)

        return val_loss

    def train(self):

        min_loss = 1e10
        for i in range(self.config.num_epochs):
            val_loss = self.train_epoch(i)

            if val_loss < min_loss:
                print(f'Generator Loss has improved from {min_loss:.4f} to {val_loss:.4f} - saving model to {self.config.checkpoint_dir}')
                min_loss = val_loss
                model_name = f'{i:03d}-generator.hdf5'
                model_path = os.path.join(self.config.checkpoint_dir, model_name)
                self.paf_gen.save(model_path)
