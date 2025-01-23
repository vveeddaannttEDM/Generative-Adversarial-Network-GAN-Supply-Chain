# Define the Generator
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model

# Define the Discriminator
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Fix Progress Printing in Training Function
def train_wgan(generator, discriminator, gan, data, epochs, batch_size, noise_dim, clip_value=0.01):
    for current_epoch in range(epochs):
        # Train Discriminator
        for _ in range(5):  # Train discriminator more frequently
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_data = generator.predict(noise)

            real_labels = -np.ones((batch_size, 1))
            fake_labels = np.ones((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_data, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # Clip discriminator weights
            for layer in discriminator.layers:
                if hasattr(layer, 'kernel'):
                    layer.kernel.assign(tf.clip_by_value(layer.kernel, -clip_value, clip_value))

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_labels = -np.ones((batch_size, 1))  # Use -1 for Wasserstein GAN

        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print Progress
        if current_epoch % 100 == 0:
            print(f"Epoch {current_epoch}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
