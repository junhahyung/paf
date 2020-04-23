import os
import tensorflow as tf
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
            loss, grads, paf = self.train_step(self.paf_gen, x, y[2], self.paf_gen_loss)
            self.paf_opt.apply_gradients(zip(grads, self.paf_gen.trainable_variables))

            losses.append(loss)

        training_loss = sum(losses) / len(losses)

        val_t = trange(self.config.validation_steps)
        val_losses = []
        for i in val_t:
            x, y = next(self.val_dataset)
            if (i == 0) & (epoch == 0):
                self.imgs_from_val = x

            loss, paf_ = self.get_loss(self.paf_gen, x, y[2], self.paf_gen_loss)

            val_losses.append(loss)
        val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch:03d}: train-loss: {training_loss:.06f} "
              f"--- val-loss: {val_loss:.06f}" )

        # tf.summary
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
            
            input_img = tf.reshape(tf.reduce_sum(input_img, -1), (1, 96, 96, 1))
            paf = tf.reshape(paf, (1, 96, 96, 2))

            tf_img = tf.concat([input_img, paf], axis=0)
            tf.summary.image('Train', tf_img, step=epoch)

            x = self.imgs_from_val
            paf = self.paf_gen(self.imgs_from_val)
            
            input_img = x[i]
            paf = paf[i]

            input_img = tf.reshape(tf.reduce_sum(input_img, -1), (1, 96, 96, 1))
            paf = tf.reshape(paf, (1, 96, 96, 2))

            tf_img = tf.concat([input_img, paf], axis=0)
            tf.summary.image('Validation', tf_img, step=epoch)

        return val_loss

    def train(self):

        min_loss = 1e10
        for i in range(self.config.num_epochs):
            val_loss = self.train_epoch(i)

            if val_loss < min_loss:
                print(f'Generator Loss has improved from {min_loss:.4f} to {val_loss:.4f} - saving model to {self.config.checkpoint_dir}')
                min_loss = val_gloss
                model_name = f'{i:03d}-generator.hdf5'
                model_path = os.path.join(self.config.checkpoint_dir, model_name)
                self.paf_gen.save(model_path)
