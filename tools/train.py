import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import argparse
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--activate_fun', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr_schedule', type=str, default='exponential')
    parser.add_argument('--decay_steps', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.9)
    parser.add_argument('--random_ratio', type=float, default=0.1)
    args = parser.parse_args()

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (args.img_size, args.img_size))
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, args.random_ratio)
        image = tf.image.random_contrast(image, 1 - args.random_ratio, 1 + args.random_ratio)
        return image, label

    (train_ds, test_ds), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds
              .map(preprocess, num_parallel_calls=AUTOTUNE)
              .map(augment, num_parallel_calls=AUTOTUNE)
              .shuffle(10000)
              .batch(args.batch_size)
              .prefetch(AUTOTUNE))

    test_ds = (test_ds
             .map(preprocess, num_parallel_calls=AUTOTUNE)
             .batch(args.test_batch_size)
             .prefetch(AUTOTUNE))

    if args.checkpoint:
        model = tf.keras.models.load_model(args.checkpoint)
    else:
        raise ValueError("No model config or checkpoint provided")

    if args.lr_schedule == 'exponential':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.initial_lr,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True
        )
    elif args.lr_schedule == 'cosine':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.initial_lr,
            decay_steps=args.decay_steps
        )
    else:
        lr_schedule = args.initial_lr

    if args.optimizer == 'adamw':
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=args.activate_fun),
        metrics=['accuracy']
    )

    log_dir = Path("logs") / f"{args.model_name}_{args.data_name}" / time.strftime('%Y%m%d-%H%M%S')
    checkpoint_dir = Path("checkpoints") / f"{args.model_name}_{args.data_name}_best"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=args.patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    return history

if __name__ == '__main__':
    main()

# python train.py --batch_size 128 --img_size 32 --optimizer adamw --lr_schedule exponential