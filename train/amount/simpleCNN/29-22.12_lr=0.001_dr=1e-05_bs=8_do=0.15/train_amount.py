import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copy
from model.simple_CNN import SimpleCNN
from amount_dataloader import getDataset


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        copy(__file__, self.filepath)
        copy("model/simple_CNN.py", self.filepath)
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("validation/loss")
        if np.less(current_loss, self.best_loss): self.best_loss = current_loss

    def on_train_end(self, logs=None):
        loss = int(self.best_loss*10000) / 10000
        try:
            os.rename(self.filepath, f"{self.filepath}_loss={loss}")
        except:
            open(f"{self.filepath}/loss={loss}.txt", "w")


class Trainer():
    def __init__(self, modelParameter, dataParameter, trainingParameter):
        super().__init__()

        lr, dr = \
            modelParameter["lr"], modelParameter["dr"]
        data_dir, val_split, iis, bs = \
            dataParameter["data_dir"], dataParameter["val_split"], dataParameter["image_size"], dataParameter["batch_size"]
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
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train/loss")
        self.val_loss_metric   = tf.keras.metrics.Mean(name="validation/loss")

        self.log = {}
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.callbacks = [
            MyCallback(filepath=self.log_dir),
            tf.keras.callbacks.EarlyStopping(
                monitor="validation/loss", mode="min",
                verbose=1, patience=patience,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath = self.log_dir + "/weights.h5", verbose=1,
                monitor="validation/loss", save_best_only=True,
                # save_weights_only=True,
            ),
        ]
        for callback in self.callbacks: callback.set_model(self.model)

        self.train_dataset, self.val_dataset = getDataset(data_dir, val_split, iis, bs)


    @tf.function
    def train_step(self, img, amount):
        with tf.GradientTape() as tape:
            logits = self.model(img, training=True)
            loss_value = self.loss_fn(amount, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_loss_metric.update_state(loss_value)
        return logits


    @tf.function
    def val_step(self, img, amount):
        logits = self.model(img, training=False)
        loss_value = self.loss_fn(amount, logits)
        self.val_loss_metric.update_state(loss_value)
        return logits


    def train(self):

        # Call 'on_train_begin()' before train begin
        for callback in self.callbacks: callback.on_train_begin(self.log)

        # Start training
        for epoch in range(self.epochs):

            desc_prefix = f"{epoch+1:3}/{self.epochs:3}"

            # Run a training loop
            pbar = tqdm(self.train_dataset, ascii=True)
            for img, amount in pbar:

                # Training steps
                self.train_step(img, amount)
                self.log[self.train_loss_metric.name] = self.train_loss_metric.result()
                train_loss = self.train_loss_metric.result()
                pbar.set_description(desc=f"{desc_prefix} | " + \
                    f"Average train loss: {train_loss:.4f}")

            # Record to TF board
            with self.writer.as_default():
                tf.summary.scalar("train/loss", train_loss, step=epoch)
            # Reset training metrics after training loop
            self.train_loss_metric.reset_states()

            # Run a validation loop
            pbar = tqdm(self.val_dataset, ascii=True)
            for img, amount in pbar:
                self.val_step(img, amount)
                self.log[self.val_loss_metric.name] = self.val_loss_metric.result()
                val_loss = self.val_loss_metric.result()
                pbar.set_description(desc=f"{desc_prefix} | " + \
                    f"Average valid loss: {val_loss:.4f}")
            
            # Record to TF board
            with self.writer.as_default():
                tf.summary.scalar("validation/loss", val_loss, step=epoch)
            # Reset validation metrics after validation loop
            self.val_loss_metric.reset_states()

            # Call 'on_epoch_end()' after every epoch
            for callback in self.callbacks: callback.on_epoch_end(epoch, self.log)
            with self.writer.as_default():
                tf.summary.scalar("learning rate", self.optimizer._decayed_lr("float32").numpy(), step=epoch)
            print("")
            if self.model.stop_training: break

        # Call 'on_train_end()' after whole training progress
        for callback in self.callbacks: callback.on_train_end(self.log)
        return


    def demo(self):

        img, amount_real = next(iter(self.val_dataset))
        amount_pred = self.model(img, training=False)

        print(f"Truth     : {amount_real}")
        print(f"Prediction: {amount_pred}")

        fig, axs = plt.subplots(2, 4)
        fig.set_size_inches(20, 10)
        fig.suptitle('Demonstration')
        for i in range(2):
            for j in range(4):
                axs[i][j].axis('off')
                axs[i][j].imshow(img[i*4+j][..., ::-1])
                # axs[i][j].set_title(f"Truth: {int(amount_real[i*4+j]*100)}% | " + \
                #                     f"Prediction: {int(amount_pred[i*4+j]*100)}%")
                axs[i][j].set_title(f"Truth: {amount_real[i*4+j]}% | " + \
                                    f"Prediction: {int(amount_pred[i*4+j])}%")
        plt.tight_layout()
        plt.savefig(self.log_dir + "/demonstration.png")
        plt.show()

        return


if __name__ == "__main__":

    dropout = 0.15
    lr = 1e-3
    dr = 1e-5
    image_size = (640, 480)  # (width, height)
    batch_size = 8
    Model = SimpleCNN(dropout=dropout)

    modelParameter = {
        "Model": Model,
        "lr"   : lr,
        "dr"   : dr,
    }
    
    dataParameter = {
        "data_dir"  : "dataset/3_192+204=396/396_spec_renamed_annotated",
        "val_split" : 0.2,
        "image_size": image_size,
        "batch_size": batch_size,
    }
    
    trainingParameter = {
        "log_dir" : f"train/amount/{Model.name}/{datetime.datetime.now().strftime(f'%d-%H.%M')}" + \
                     f"_lr={lr}_dr={dr}_bs={batch_size}_do={dropout}",
        "epochs"  : 300,
        "patience": 15,
    }
    
    trainer = Trainer(modelParameter, dataParameter, trainingParameter)
    trainer.train()
    trainer.demo()

    pass