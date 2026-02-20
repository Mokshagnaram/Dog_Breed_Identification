import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrainConfig:
    dataset_dir: str = "dataset"
    train_dir_name: str = "train"
    test_dir_name: str = "test"
    image_size: tuple = (160, 160)
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 30
    model_output_path: str = "model/dog_breed_model.h5"
    plot_output_path: str = "model/training_curves.png"
    tfds_dataset_name: str = "stanford_dogs"
    tfds_data_dir: str = "tfds_data"
    force_rebuild_dataset: bool = False
    demo_breed_count: int = 5
    demo_images_per_breed: int = 200


class DogBreedTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.train_dir = os.path.join(self.config.dataset_dir, self.config.train_dir_name)
        self.test_dir = os.path.join(self.config.dataset_dir, self.config.test_dir_name)
        self.prepared_marker = os.path.join(self.config.dataset_dir, ".tfds_prepared")
        self.effective_batch_size = self._select_batch_size()
        self.model = None

    def _select_batch_size(self) -> int:
        # Use the largest configured batch size (minimum 32) for higher throughput.
        return max(32, self.config.batch_size)

    @staticmethod
    def _has_images(root_dir: str) -> bool:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if not os.path.isdir(root_dir):
            return False
        for path in Path(root_dir).rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_exts:
                return True
        return False

    def _save_split_to_directory(self, split_name: str, dataset_split, label_names, target_root: str) -> int:
        count = 0
        os.makedirs(target_root, exist_ok=True)

        for image_tensor, label_tensor in tfds.as_numpy(dataset_split):
            class_index = int(label_tensor)
            class_name = label_names[class_index]
            class_dir = os.path.join(target_root, class_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{split_name}_{count:06d}.jpg"
            save_path = os.path.join(class_dir, filename)
            tf.keras.utils.save_img(save_path, image_tensor)
            count += 1

        return count

    def _reset_export_directories(self) -> None:
        for export_dir in (self.train_dir, self.test_dir):
            if os.path.isdir(export_dir):
                shutil.rmtree(export_dir)
            os.makedirs(export_dir, exist_ok=True)

    def _write_marker(self, source_name: str, train_count: int, test_count: int) -> None:
        with open(self.prepared_marker, "w", encoding="utf-8") as marker_file:
            marker_file.write(
                f"dataset={source_name}\n"
                f"train_images={train_count}\n"
                f"test_images={test_count}\n"
            )

    def _prepare_dataset_from_stanford_tfds(self) -> None:
        print(f"Preparing dataset from TFDS: {self.config.tfds_dataset_name}")
        print("This will download automatically on first run and reuse local cache later.")

        builder = tfds.builder(self.config.tfds_dataset_name, data_dir=self.config.tfds_data_dir)
        builder.download_and_prepare()

        label_names = builder.info.features["label"].names
        train_split = builder.as_dataset(split="train", as_supervised=True)
        test_split = builder.as_dataset(split="test", as_supervised=True)

        self._reset_export_directories()

        train_count = self._save_split_to_directory(
            split_name="train",
            dataset_split=train_split,
            label_names=label_names,
            target_root=self.train_dir,
        )
        test_count = self._save_split_to_directory(
            split_name="test",
            dataset_split=test_split,
            label_names=label_names,
            target_root=self.test_dir,
        )

        self._write_marker(self.config.tfds_dataset_name, train_count, test_count)
        print(f"Dataset export completed: {train_count} train images, {test_count} test images.")

    def _save_cifar_dog_subset(self, split_name: str, images: np.ndarray, target_root: str) -> int:
        breed_names = [f"cifar_demo_breed_{idx + 1}" for idx in range(self.config.demo_breed_count)]
        count = 0

        for image in images:
            breed_name = breed_names[count % self.config.demo_breed_count]
            class_dir = os.path.join(target_root, breed_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{split_name}_{count:06d}.jpg"
            save_path = os.path.join(class_dir, filename)
            tf.keras.utils.save_img(save_path, image)
            count += 1

        return count

    def _prepare_dataset_from_cifar10(self) -> None:
        print("Falling back to CIFAR-10. Filtering dog class and creating demo breed folders.")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        dog_label = 5
        train_dogs = x_train[y_train.flatten() == dog_label]
        test_dogs = x_test[y_test.flatten() == dog_label]

        if len(train_dogs) == 0 or len(test_dogs) == 0:
            raise ValueError("CIFAR-10 dog subset is empty.")

        self._reset_export_directories()
        train_count = self._save_cifar_dog_subset("train", train_dogs, self.train_dir)
        test_count = self._save_cifar_dog_subset("test", test_dogs, self.test_dir)

        self._write_marker("cifar10_dog_demo", train_count, test_count)
        print(f"CIFAR-10 demo dataset export completed: {train_count} train images, {test_count} test images.")

    def _prepare_local_synthetic_dataset(self) -> None:
        print("CIFAR-10 fallback also failed. Generating local synthetic images to keep pipeline runnable.")
        self._reset_export_directories()

        rng = np.random.default_rng(42)
        train_total = self.config.demo_breed_count * self.config.demo_images_per_breed
        test_total = max(1, self.config.demo_breed_count * (self.config.demo_images_per_breed // 5))

        for split_name, total_count, target_root in (
            ("train", train_total, self.train_dir),
            ("test", test_total, self.test_dir),
        ):
            for index in range(total_count):
                breed_name = f"synthetic_breed_{(index % self.config.demo_breed_count) + 1}"
                class_dir = os.path.join(target_root, breed_name)
                os.makedirs(class_dir, exist_ok=True)

                image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
                save_path = os.path.join(class_dir, f"{split_name}_{index:06d}.jpg")
                tf.keras.utils.save_img(save_path, image)

        self._write_marker("synthetic_demo", train_total, test_total)
        print(f"Synthetic dataset export completed: {train_total} train images, {test_total} test images.")

    def prepare_dataset(self) -> None:
        train_ready = self._has_images(self.train_dir)
        test_ready = self._has_images(self.test_dir)

        # If local dataset folders are already populated, use them directly.
        if train_ready and test_ready and not self.config.force_rebuild_dataset:
            print("Dataset already prepared in dataset/train and dataset/test. Skipping dataset export.")
            return

        try:
            self._prepare_dataset_from_stanford_tfds()
            return
        except Exception as exc:
            print(f"Stanford Dogs download/preparation failed: {exc}")

        try:
            self._prepare_dataset_from_cifar10()
            return
        except Exception as exc:
            print(f"CIFAR-10 fallback failed: {exc}")

        self._prepare_local_synthetic_dataset()

    def validate_directories(self) -> None:
        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        if not os.path.isdir(self.test_dir):
            raise FileNotFoundError(f"Validation/Test directory not found: {self.test_dir}")
        if not self._has_images(self.train_dir):
            raise FileNotFoundError(f"No images found in training directory: {self.train_dir}")
        if not self._has_images(self.test_dir):
            raise FileNotFoundError(f"No images found in validation/test directory: {self.test_dir}")

    def create_generators(self):
        # Training data generator with augmentation.
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        # Validation data generator with only rescaling.
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.config.image_size,
            batch_size=self.effective_batch_size,
            class_mode="categorical",
            shuffle=True,
        )

        val_generator = val_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.config.image_size,
            batch_size=self.effective_batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        return train_generator, val_generator

    def create_tf_data_pipeline(self, train_generator, val_generator, num_classes: int):
        image_signature = tf.TensorSpec(
            shape=(None, self.config.image_size[0], self.config.image_size[1], 3),
            dtype=tf.float32,
        )
        label_signature = tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        output_signature = (image_signature, label_signature)

        train_steps = len(train_generator)
        val_steps = len(val_generator)

        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_generator, output_signature=output_signature
        )
        val_dataset = tf.data.Dataset.from_generator(
            lambda: val_generator, output_signature=output_signature
        )

        cache_tag = (
            f"{num_classes}_{self.config.image_size[0]}x{self.config.image_size[1]}"
            f"_b{self.effective_batch_size}_{os.getpid()}"
        )
        train_cache_path = os.path.join(self.config.dataset_dir, f".train_cache_{cache_tag}")
        val_cache_path = os.path.join(self.config.dataset_dir, f".val_cache_{cache_tag}")

        # Remove stale cache files for this configuration to avoid schema mismatch.
        for cache_path in (train_cache_path, val_cache_path):
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except OSError:
                pass

        train_dataset = (
            train_dataset.take(train_steps)
            .cache(train_cache_path)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            val_dataset.take(val_steps)
            .cache(val_cache_path)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_dataset, val_dataset

    def build_model(self, num_classes: int) -> Model:
        try:
            base_model = VGG19(
                weights="imagenet",
                include_top=False,
                input_shape=(160, 160, 3),
            )
            print("Loaded VGG19 with ImageNet pretrained weights.")
        except Exception as exc:
            print(f"Could not load pretrained VGG19 weights: {exc}")
            print("Falling back to randomly initialized VGG19 weights for offline training.")
            base_model = VGG19(
                weights=None,
                include_top=False,
                input_shape=(160, 160, 3),
            )

        # Freeze all VGG19 convolutional layers.
        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def get_callbacks(self):
        os.makedirs(os.path.dirname(self.config.model_output_path), exist_ok=True)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )

        checkpoint = ModelCheckpoint(
            filepath=self.config.model_output_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        return [early_stopping, checkpoint]

    def plot_training_curves(self, history) -> None:
        acc = history.history.get("accuracy", [])
        val_acc = history.history.get("val_accuracy", [])
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(acc, label="Train Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.config.plot_output_path)
        plt.close()

    def train(self) -> None:
        self.prepare_dataset()
        self.validate_directories()

        train_generator, val_generator = self.create_generators()
        num_classes = len(train_generator.class_indices)

        if num_classes < 2:
            raise ValueError(
                "At least two classes are required for training. "
                "Please check your dataset structure under dataset/train/."
            )

        self.model = self.build_model(num_classes=num_classes)

        print("\nClass mapping:")
        for class_name, class_index in train_generator.class_indices.items():
            print(f"{class_index}: {class_name}")

        callbacks = self.get_callbacks()

        train_dataset, val_dataset = self.create_tf_data_pipeline(
            train_generator=train_generator,
            val_generator=val_generator,
            num_classes=num_classes,
        )

        fit_kwargs = dict(
            x=train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator),
            verbose=1,
            workers=4,
            use_multiprocessing=True,
        )

        try:
            history = self.model.fit(**fit_kwargs)
        except TypeError:
            # Compatibility fallback for runtimes that do not expose workers/use_multiprocessing.
            fit_kwargs.pop("workers", None)
            fit_kwargs.pop("use_multiprocessing", None)
            history = self.model.fit(**fit_kwargs)

        # Save final model as well (best model is already saved by checkpoint).
        self.model.save(self.config.model_output_path)
        print(f"\nModel saved to: {self.config.model_output_path}")

        try:
            self.plot_training_curves(history)
            print(f"Training curves saved to: {self.config.plot_output_path}")
        except Exception as exc:
            print(f"Warning: failed to save training curves: {exc}")


def main() -> None:
    # Print detected devices and configure GPU memory growth if available.
    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")

    print(f"Detected CPUs: {cpus}")
    print(f"Detected GPUs: {gpus}")
    if gpus:
        print("GPU acceleration is enabled via TensorFlow device placement.")
    else:
        print("No GPU detected. Training will run on CPU fallback.")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    config = TrainConfig()
    trainer = DogBreedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
