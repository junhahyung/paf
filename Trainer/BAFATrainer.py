import os
import tensorflow as tf
from tqdm import trange
from utils.losses import *

"""
Boundary-Aware Face Alignment Trainer
"""
class Trainer():
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.config = config
        self.callbacks = list()

        # init model parts
        self.generator = self.model.generator
        self.generator_loss = weighted_cross_entropy(beta=10)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.000005, 4 * self.config.training_steps, 0.90, staircase=True)
        self.g_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.regressor = self.model.regressor
        self.regressor_loss = dist
        r_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 2 * self.config.training_steps, 0.95, staircase=True)
        self.r_opt = tf.keras.optimizers.Adam(learning_rate=r_lr_schedule)

        self.discriminator = self.model.discriminator
        self.gen_and_disc = self.model.gen_and_disc

        def wasserstein_loss(y_true, y_pred):
            return tf.reduce_mean(y_true * y_pred)

        self.discriminator_loss = wasserstein_loss #tf.keras.losses.BinaryCrossentropy()
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.gd_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        #init dataset
        self.dataset = iter(train_data())
        self.val_dataset = iter(val_data())

        # init tensorboard
        self.tb_writer = tf.summary.create_file_writer(self.config.tensorboard_log_dir, name='train')
        self.val_tb_writer = tf.summary.create_file_writer(self.config.tensorboard_log_dir, name='val')

        self.imgs_from_train = None
        self.imgs_from_val = None

        if self.config.finetune_regressor:
            self.load_weight()

    def load_weight(self):
        model_list = os.listdir(os.path.join('experiments/', self.config.pretrained_exp, 'checkpoints'))
        model_list.sort()

        generator_model_path = model_list[-2]
        self.model.generator.load_weights(os.path.join('experiments/', self.config.pretrained_exp, 'checkpoints', generator_model_path))
        print(f"Loaded pretrained {self.config.pretrained_exp} generator weights.")

    def get_loss(self, model, x, y, loss_function, training=False):
        y_ = model(x, training=training)
        return loss_function(y_true=y, y_pred=y_), y_

    @tf.function()
    def train_step(self, model, x, y, loss_function):
        with tf.GradientTape() as tape:
            loss_value, y_ = self.get_loss(model, x, y, loss_function, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), y_

    def pretrain_critics(self, steps=500):
        print(f"[*] Pretraining critics for {steps} steps...")

        t = trange(steps)

        for i in t:
            x, y = next(self.dataset)
            # train discriminator
            dloss_real, grads, logit_ = self.train_step(self.discriminator, y[0], tf.ones([self.config.batch_size, 1]), self.discriminator_loss)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

    def train_epoch(self, epoch):
        t = trange(self.config.training_steps)
        glosses = []
        rlosses = []
        dlosses_real = []
        dlosses_fake = []

        for i in t:
            x, y = next(self.dataset)

            if (i == 0) & (epoch == 0):
                self.imgs_from_train = x

            # if self.config.finetune_regressor:
            #     gloss, bm_ = self.get_loss(self.generator, x, y[0], self.generator_loss, training=False)
            # else:

            # train generator
            gloss, grads, bm_ = self.train_step(self.generator, x, y[0], self.generator_loss)
            self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

            # train regressor
            rloss, grads, lm_ = self.train_step(self.regressor, [x, bm_], y[1], self.regressor_loss)
            self.r_opt.apply_gradients(zip(grads, self.regressor.trainable_variables))

            # train discriminator
            dloss_real, grads, logit_ = self.train_step(self.discriminator, y[0], tf.ones([self.config.batch_size, 1]), self.discriminator_loss)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

            # train generator to trick discriminator
            dloss_fake, grads, logit_ = self.train_step(self.gen_and_disc, x, tf.ones([self.config.batch_size, 1]) * -1.0, self.discriminator_loss)
            self.gd_opt.apply_gradients(zip(grads, self.gen_and_disc.trainable_variables))

            glosses.append(gloss)
            rlosses.append(rloss)
            dlosses_real.append(dloss_real)
            dlosses_fake.append(dloss_fake)

        training_gloss = sum(glosses) / len(glosses)
        training_rloss = sum(rlosses) / len(rlosses)

        val_t = trange(self.config.validation_steps)
        val_glosses = []
        val_rlosses = []
        for i in val_t:
            x, y = next(self.val_dataset)
            if (i == 0) & (epoch == 0):
                self.imgs_from_val = x

            gloss, bm_ = self.get_loss(self.generator, x, y[0], self.generator_loss)
            rloss, lm_ = self.get_loss(self.regressor, [x, bm_], y[1], self.regressor_loss)

            val_glosses.append(gloss)
            val_rlosses.append(rloss)
        val_gloss = sum(val_glosses) / len(val_glosses)
        val_rloss = sum(val_rlosses) / len(val_rlosses)

        print(f"Epoch {epoch:03d}: train-gloss: {training_gloss:.06f} --- train-rloss: {training_rloss:.06f} "
              f"--- val-gloss: {val_gloss:.06f} --- val-rloss: {val_rloss:.06f}")

        # tf.summary
        with self.tb_writer.as_default():
            # scalar
            tf.summary.scalar('Binary Cross Entropy', training_gloss, step=epoch)
            tf.summary.scalar('MSE', training_rloss, step=epoch)
            tf.summary.scalar('Val Binary Cross Entropy', val_gloss, step=epoch)
            tf.summary.scalar('Val MSE', val_rloss, step=epoch)

            tf.summary.scalar('Real Critic Loss', sum(dlosses_real) / len(dlosses_real), step=epoch)
            tf.summary.scalar('Fake Critic Loss', sum(dlosses_fake) / len(dlosses_fake), step=epoch)

            # record images
            i = 0
            x = self.imgs_from_train
            bm_ = self.generator(self.imgs_from_train)
            lm_ = self.regressor([x, bm_])

            input_img = x[i]
            boundary_map = tf.reduce_sum(bm_[i], -1)
            landmark = tf.scatter_nd(tf.cast(lm_[i], tf.int32), [1.0] * 60,  (96, 96))
            landmark = tf.reshape(landmark, (1, 96, 96, 1))
            landmark = tf.image.rot90(tf.image.flip_up_down(landmark), 3)

            input_img = tf.reshape(tf.reduce_sum(input_img, -1), (1, 96, 96, 1))
            boundary_map = tf.reshape(boundary_map, (1, 96, 96, 1))
            landmark = tf.reshape(landmark, (1, 96, 96, 1))

            tf_img = tf.concat([input_img, boundary_map, landmark], axis=0)
            tf.summary.image('Train', tf_img, step=epoch)

            x = self.imgs_from_val
            bm_ = self.generator(x)
            lm_ = self.regressor([x, bm_])

            input_img = x[i]
            boundary_map = tf.reduce_sum(bm_[i], -1)
            landmark = tf.scatter_nd(tf.cast(lm_[i], tf.int32), [1.0] * 60,  (96, 96))
            landmark = tf.reshape(landmark, (1, 96, 96, 1))
            landmark = tf.image.rot90(tf.image.flip_up_down(landmark), 3)

            input_img = tf.reshape(tf.reduce_sum(input_img, -1), (1, 96, 96, 1))
            boundary_map = tf.reshape(boundary_map, (1, 96, 96, 1))
            landmark = tf.reshape(landmark, (1, 96, 96, 1))

            tf_img = tf.concat([input_img, boundary_map, landmark], axis=0)
            tf.summary.image('Validation', tf_img, step=epoch)

        return val_gloss, val_rloss

    def train(self):

        # self.pretrain_critics()

        min_gloss = 1e10
        min_rloss = 1e10
        for i in range(self.config.num_epochs):
            val_gloss, val_rloss = self.train_epoch(i)

            if val_gloss < min_gloss:
                print(f'Generator Loss has improved from {min_gloss:.4f} to {val_gloss:.4f} - saving model to {self.config.checkpoint_dir}')
                min_gloss = val_gloss
                model_name = f'{i:03d}-generator.hdf5'
                model_path = os.path.join(self.config.checkpoint_dir, model_name)
                self.generator.save(model_path)

            if val_rloss < min_rloss:
                print(f'Regressor Loss has improved from {min_rloss:.4f} to {val_rloss:.4f} - saving model to {self.config.checkpoint_dir}')
                min_rloss = val_rloss
                model_name = f'{i:03d}-regressor.hdf5'
                model_path = os.path.join(self.config.checkpoint_dir, model_name)
                self.regressor.save(model_path)
