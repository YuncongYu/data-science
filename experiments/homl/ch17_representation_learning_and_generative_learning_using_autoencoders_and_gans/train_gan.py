import numpy as np
import tensorflow as tf
import tqdm


def train_gan(
    gan: tf.keras.models.Model, x_train: np.ndarray, coding_size: int, batch_size: int, n_epochs: int = 50
) -> None:
    batch_size = 32
    ds_train = tf.data.Dataset.from_tensor_slices(x_train.astype(np.float32)).shuffle(1000)
    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True).prefetch(1)

    generator, discriminator = gan.layers

    for epoch in tqdm.tqdm(range(n_epochs), desc="Epoch", total=n_epochs):
        for x_real in tqdm.tqdm(ds_train, desc="Batch"):
            # Phase 1: train discriminator
            codings = tf.random.normal(shape=(batch_size, coding_size))
            x_fake = generator(codings)
            x_real_and_fake = tf.concat([x_real, x_fake], axis=0)
            y_real_and_fake = tf.constant([[1.0]] * batch_size + [[0.0]] * batch_size)

            discriminator.trainable = True
            discriminator.train_on_batch(x_real_and_fake, y_real_and_fake)

            # Phase 2: train generator
            codings = tf.random.normal(shape=(batch_size, coding_size))
            y_counterfeit = tf.constant([[1.0]] * batch_size)

            discriminator.trainable = False
            gan.train_on_batch(codings, y_counterfeit)
