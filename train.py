import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copy
from model.Unet import Unet, MyUnet
from model.simple_CNN import SimpleCNN
from dataloader import getDataset


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        copy(__file__, self.filepath)
        copy("unet.py", self.filepath)
        self.best_loss = np.Inf
        self.best_acc  = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("validation/loss")
        if np.less(current_loss, self.best_loss): self.best_loss = current_loss
        current_acc  = logs.get("validation/accuracy")
        if np.greater(current_acc, self.best_acc): self.best_acc = current_acc

    def on_train_end(self, logs=None):
        loss = int(self.best_loss*10000) / 10000
        acc  = int(self.best_acc*10000) / 100
        try:
            os.rename(self.filepath, f"{self.filepath}_acc={acc}%_loss={loss}")
        except:
            with open(f"{self.filepath}/acc={acc}%_loss={loss}.txt", "w") as _: pass


class Trainer():
    def __init__(self, modelParameter, dataParameter, trainingParameter):
        super().__init__()

        lr, dr = \
            modelParameter["lr"], modelParameter["dr"]
        data_dir, classes, iis, bs, mt = \
            dataParameter["data_dir"], dataParameter["classes"], dataParameter["image_size"], dataParameter["batch_size"], dataParameter["mask_type"]
        self.log_dir, self.epochs, patience = \
            trainingParameter["log_dir"], trainingParameter["epochs"], trainingParameter["patience"]
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=dr)

        self.model = modelParameter["Model"]
        self.model.build(input_shape=(bs, iis[0], iis[1], 3))
        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train/loss")
        self.train_acc_metric  = tf.keras.metrics.SparseCategoricalAccuracy(name="train/accuracy")
        self.val_loss_metric   = tf.keras.metrics.Mean(name="validation/loss")
        self.val_acc_metric    = tf.keras.metrics.SparseCategoricalAccuracy(name="validation/accuracy")

        self.log = {}
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.callbacks = [
            MyCallback(filepath=self.log_dir),
            tf.keras.callbacks.EarlyStopping(
                monitor="validation/loss", mode="min",
                verbose=1, patience=patience,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath = self.log_dir + "/val_loss.h5", verbose=1,
                monitor="validation/loss", save_best_only=True,
                # save_weights_only=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath = self.log_dir + "/val_acc.h5", verbose=1,
                monitor="validation/accuracy", save_best_only=True,
                # save_weights_only=True,
            ),
        ]
        for callback in self.callbacks: callback.set_model(self.model)

        self.train_dataset, self.val_dataset = getDataset(data_dir, classes, iis, bs, mt)


    @tf.function
    def train_step(self, img, status, mask=None):
        with tf.GradientTape() as tape:
            logits = self.model(img, training=True)
            loss_value = self.loss_fn(status, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_loss_metric.update_state(loss_value)
        self.train_acc_metric.update_state(status, logits)
        return logits


    @tf.function
    def val_step(self, img, status, mask=None):
        logits = self.model(img, training=False)
        loss_value = self.loss_fn(status, logits)
        self.val_loss_metric.update_state(loss_value)
        self.val_acc_metric.update_state(status, logits)
        return logits


    def train(self):

        # Call 'on_train_begin()' before train begin
        for callback in self.callbacks: callback.on_train_begin(self.log)

        # Start training
        for epoch in range(self.epochs):

            desc_prefix = f"{epoch+1:3}/{self.epochs:3}"

            # Run a training loop
            pbar = tqdm(self.train_dataset, ascii=True)
            for imgs, statuses, masks in pbar:

                # Training steps
                self.train_step(imgs, statuses)
                for m in [self.train_loss_metric, self.train_acc_metric]:
                    self.log[m.name] = m.result()
                train_loss = self.train_loss_metric.result()
                train_acc  = self.train_acc_metric.result()
                pbar.set_description(desc=f"{desc_prefix} | " + \
                    f"Average train loss: {train_loss:.4f} | " + \
                    f"Average train acc: {train_acc:.4f}")

            # Record to TF board
            with self.writer.as_default():
                tf.summary.scalar("train/loss"    , train_loss, step=epoch)
                tf.summary.scalar("train/accuracy", train_acc , step=epoch)
            # Reset training metrics after training loop
            self.train_loss_metric.reset_states()
            self.train_acc_metric.reset_states()

            # Run a validation loop
            pbar = tqdm(self.val_dataset, ascii=True)
            for imgs, statuses, masks in pbar:
                self.val_step(imgs, statuses)
                for m in [self.val_loss_metric, self.val_acc_metric]:
                    self.log[m.name] = m.result()
                val_loss = self.val_loss_metric.result()
                val_acc  = self.val_acc_metric.result()
                pbar.set_description(desc=f"{desc_prefix} | " + \
                    f"Average valid loss: {val_loss:.4f} | " + \
                    f"Average valid acc: {val_acc:.4f}")
            
            # Record to TF board
            with self.writer.as_default():
                tf.summary.scalar("validation/loss"    , val_loss, step=epoch)
                tf.summary.scalar("validation/accuracy", val_acc , step=epoch)
            # Reset validation metrics after validation loop
            self.val_loss_metric.reset_states()
            self.val_acc_metric.reset_states()

            # Call 'on_epoch_end()' after every epoch
            for callback in self.callbacks: callback.on_epoch_end(epoch, self.log)
            with self.writer.as_default():
                tf.summary.scalar('learning rate', self.optimizer._decayed_lr('float32').numpy(), step=epoch)
            print("")
            if self.model.stop_training: break

        # Call 'on_train_end()' after whole training progress
        for callback in self.callbacks: callback.on_train_end(self.log)
        return


    def demo(self):

        imgs, statuses, masks = next(iter(self.val_dataset))
        prediction = self.model(imgs, training=False)

        fig, axs = plt.subplots(2, 4)
        fig.set_size_inches(20, 10)
        fig.suptitle('Demonstration')
        for i in range(2):
            for j in range(4):
                axs[i][j].axis('off')
                axs[i][j].imshow(imgs[i*2+j])
                axs[i][j].set_title(f"Ans: {statuses[i*2+j]} | " + \
                    f"Pred: {[ f'{p:.3f}' for p in prediction[i*2+j]]}")
        plt.tight_layout()
        plt.savefig(self.log_dir + "/demonstration.png")
        plt.show()

        return


if __name__ == "__main__":

    dropout = 0.9
    lr = 9e-4
    dr = 1e-10
    image_size = (480, 640)
    batch_size = 8
    Model = SimpleCNN(dropout=dropout)

    modelParameter = {
        "Model": Model,
        "lr"   : lr,
        "dr"   : dr,
    }
    
    dataParameter = {
        "data_dir"    : "dataset/125_8-2_classified",
        "classes"   : ["bag of vegetable", "bell pepper", "box of vegetable", "corn", "daikon", "grape", "orange", "potato", "tomato"],
        "image_size": image_size,
        "batch_size": batch_size,
        "mask_type" : "normal",
    }
    
    trainingParameter = {
        "log_dir" : f"train/class/{Model.name}/{datetime.datetime.now().strftime(f'%d-%H.%M')}" + \
                     f"_lr={lr}_dr={dr}_bs={batch_size}_do={dropout}",
        "epochs"  : 300,
        "patience": 15,
    }
    
    trainer = Trainer(modelParameter, dataParameter, trainingParameter)
    trainer.train()
    trainer.demo()

    pass