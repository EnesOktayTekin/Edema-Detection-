# -*- coding: utf-8 -*-
"""
# VGG16 Transfer Learning for Binary Image Classification
A comprehensive implementation using TensorFlow/Keras with various training approaches.

This script demonstrates how to use VGG16 pre-trained model for binary image classification
through transfer learning with two different approaches:
1. Feature extraction (frozen base model)
2. Fine-tuning (partially trainable base model)

Author: Enes Oktay Tekin
Date: June 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import cv2

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Directory Setup
# Replace with your own data paths
DATA_DIR = './img_data/'
train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'val')
test_dir = os.path.join(DATA_DIR, 'test')

def create_data_generators(img_size=(224, 224), batch_size=20):
    """
    Create and return data generators for training, validation, and testing.
    
    Args:
        img_size (tuple): Target image dimensions
        batch_size (int): Number of samples per batch
    
    Returns:
        tuple: Training, validation, and test data generators
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation and test data generators (no augmentation, only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators from directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    test_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def build_vgg16_model(trainable_from_block5=False, use_regularization=False):
    """
    Build a transfer learning model based on VGG16.
    
    Args:
        trainable_from_block5 (bool): If True, make block5 layers trainable 
        use_regularization (bool): If True, add dropout and L2 regularization
        
    Returns:
        tf.keras.Model: Compiled transfer learning model
    """
    # Load pre-trained VGG16 model without top layers
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Configure which layers to make trainable
    conv_base.trainable = trainable_from_block5
    if trainable_from_block5:
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    
    # Create sequential model
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    
    # Add different classifier layers based on regularization flag
    if use_regularization:
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
    else:
        model.add(Dense(256, activation='relu'))
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with appropriate optimizer
    if trainable_from_block5:
        model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(learning_rate=1e-5),  # Lower learning rate when fine-tuning
            metrics=['acc']
        )
    else:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=1e-4),
            metrics=['acc']
        )
    
    # Display model summary
    model.summary()
    
    return model

def train_model(model, train_generator, validation_generator, 
                epochs=30, steps_per_epoch=35, use_callbacks=True, model_name='model'):
    """
    Train the model with optional callbacks.
    
    Args:
        model (tf.keras.Model): The model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs (int): Number of training epochs
        steps_per_epoch (int): Steps per epoch
        use_callbacks (bool): Whether to use learning rate and early stopping callbacks
        model_name (str): Name for saving the model
        
    Returns:
        history: Training history
    """
    callbacks = []
    if use_callbacks:
        # Dynamic learning rate adjustment
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        callbacks = [reduce_lr, early_stopping]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )
    
    # Save the trained model
    model.save(f'{model_name}.h5')
    
    return history

def evaluate_model(model, test_generator):
    """
    Evaluate model performance on test data.
    
    Args:
        model (tf.keras.Model): Trained model
        test_generator: Test data generator
        
    Returns:
        tuple: Test loss and accuracy
    """
    # Ensure steps are correctly calculated for evaluation
    test_steps = test_generator.samples // test_generator.batch_size
    if test_generator.samples % test_generator.batch_size != 0:
        test_steps += 1
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    return test_loss, test_acc

def calculate_metrics(model, test_generator):
    """
    Calculate and print precision, recall, F1 score, and confusion matrix.
    
    Args:
        model (tf.keras.Model): Trained model
        test_generator: Test data generator
        
    Returns:
        tuple: Precision, recall, and F1 score
    """
    # Calculate steps for prediction
    test_steps = test_generator.samples // test_generator.batch_size
    if test_generator.samples % test_generator.batch_size != 0:
        test_steps += 1
    
    # Make predictions
    predictions = model.predict(test_generator, steps=test_steps)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    true_labels = test_generator.classes
    
    # Ensure prediction array matches the length of true labels
    predicted_labels = predicted_labels[:len(true_labels)]
    
    # Calculate metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Print results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Get class name mapping
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    target_names = [class_names.get(0, "class_0"), class_names.get(1, "class_1")]
    
    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))
    
    return precision, recall, f1

def visualize_predictions(model, test_generator, num_images=5):
    """
    Visualize model predictions on test images.
    
    Args:
        model (tf.keras.Model): Trained model
        test_generator: Test data generator
        num_images (int): Number of images to visualize
    """
    # Get a batch of test images and labels
    test_images, test_labels = next(test_generator)
    
    # Get class name mapping
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Make predictions
    predictions = model.predict(test_images)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Visualize
    plt.figure(figsize=(20, 4 * min(num_images, len(test_images))))
    for i in range(min(num_images, len(test_images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(test_images[i])
        
        # Get class names
        true_class_name = class_names.get(int(test_labels[i]), "Unknown")
        pred_class_name = class_names.get(predicted_classes[i], "Unknown")
        
        plt.title(f"True: {true_class_name}\nPred: {pred_class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_grad_cam(model, img_array, layer_name='block5_conv3'):
    """
    Generate Grad-CAM heatmap for model interpretation.
    
    Args:
        model (tf.keras.Model): Trained model
        img_array (numpy.ndarray): Input image array
        layer_name (str): Name of the layer to use for Grad-CAM
        
    Returns:
        numpy.ndarray: Heatmap
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer and output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    # Extract features
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Guided backpropagation
    gate_f = tf.cast(output > 0, "float32")
    gate_r = tf.cast(grads > 0, "float32")
    guided_grads = gate_f * gate_r * grads
    
    # Average gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    # Create class activation map
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    # Process CAM
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
    
    return heatmap

def visualize_with_grad_cam(model, test_generator, num_images=5):
    """
    Visualize images with Grad-CAM heatmaps overlaid.
    
    Args:
        model (tf.keras.Model): Trained model
        test_generator: Test data generator
        num_images (int): Number of images to visualize
    """
    # Get a batch of test images and true labels
    test_images, test_labels = next(test_generator)
    
    # Get class mapping
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Make predictions
    predictions = model.predict(test_images)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Find correct predictions
    correct_pred = predicted_classes == test_labels.astype(int)
    correct_indices = np.where(correct_pred)[0]
    
    # Select random correct predictions to visualize
    if len(correct_indices) > 0:
        indices = np.random.choice(correct_indices, 
                                   min(num_images, len(correct_indices)), 
                                   replace=False)
        
        plt.figure(figsize=(20, 8 * min(num_images, len(indices))))
        
        for i, idx in enumerate(indices):
            img = test_images[idx]
            img_array = np.expand_dims(img, axis=0)
            
            # Generate heatmap
            heatmap = get_grad_cam(model, img_array)
            
            # Convert heatmap to RGB
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Superimpose heatmap on original image
            orig_img = np.uint8(img * 255)
            superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
            
            # Plot original image
            plt.subplot(num_images, 2, 2*i + 1)
            plt.imshow(img)
            plt.title(f"Original Image\nClass: {class_names.get(int(test_labels[idx]), 'Unknown')}")
            plt.axis('off')
            
            # Plot heatmap
            plt.subplot(num_images, 2, 2*i + 2)
            plt.imshow(superimposed)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No correct predictions found in this batch.")

def compare_models(models_metrics, model_names):
    """
    Compare different models based on their metrics.
    
    Args:
        models_metrics (list): List of (precision, recall, f1) tuples for each model
        model_names (list): List of model names
    """
    # Extract metrics
    precision_scores = [metrics[0] for metrics in models_metrics]
    recall_scores = [metrics[1] for metrics in models_metrics]
    f1_scores = [metrics[2] for metrics in models_metrics]
    
    # Set up bar chart
    labels = ['Precision', 'Recall', 'F1 Score']
    x = np.arange(len(labels))
    width = 0.8 / len(model_names)  # Width of the bars
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        ax.bar(x + i*width - (len(model_names)-1)*width/2, 
              [precision_scores[i], recall_scores[i], f1_scores[i]], 
              width, 
              label=model_name)
    
    # Add labels and title
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    
    # Add score values above bars
    for i, model_name in enumerate(model_names):
        metrics = [precision_scores[i], recall_scores[i], f1_scores[i]]
        for j, v in enumerate(metrics):
            ax.text(j + i*width - (len(model_names)-1)*width/2, 
                   v + 0.01, 
                   f'{v:.2f}', 
                   ha='center')
    
    # Improve layout
    plt.ylim(0, 1.0)
    fig.tight_layout()
    plt.show()

def plot_training_history(history, metrics=['acc', 'loss']):
    """
    Plot training history metrics.
    
    Args:
        history: Training history object
        metrics (list): Metrics to plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        axes[i].plot(history.history[metric], label=f'Training {metric}')
        axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        axes[i].set_title(f'Model {metric}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the complete pipeline.
    """
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Approach 1: Feature extraction (frozen base model)
    print("\n--- Training Model 1: Feature extraction (frozen VGG16) ---")
    model1 = build_vgg16_model(trainable_from_block5=False, use_regularization=False)
    history1 = train_model(
        model1, 
        train_generator, 
        validation_generator, 
        epochs=50, 
        steps_per_epoch=20, 
        model_name='model1_feature_extraction'
    )
    
    # Approach 2: Fine-tuning (unfreeze some layers)
    print("\n--- Training Model 2: Fine-tuning with regularization ---")
    model2 = build_vgg16_model(trainable_from_block5=True, use_regularization=True)
    history2 = train_model(
        model2, 
        train_generator, 
        validation_generator, 
        epochs=30, 
        steps_per_epoch=35, 
        model_name='model2_fine_tuning'
    )
    
    # Evaluate models
    print("\n--- Evaluating Model 1 ---")
    evaluate_model(model1, test_generator)
    metrics1 = calculate_metrics(model1, test_generator)
    
    print("\n--- Evaluating Model 2 ---")
    evaluate_model(model2, test_generator)
    metrics2 = calculate_metrics(model2, test_generator)
    
    # Compare models
    compare_models(
        [metrics1, metrics2],
        ['Feature Extraction', 'Fine-Tuning with Regularization']
    )
    
    # Plot training history
    print("\n--- Plotting Training History ---")
    plot_training_history(history1)
    plot_training_history(history2)
    
    # Visualize predictions
    print("\n--- Visualizing Model Predictions ---")
    visualize_predictions(model2, test_generator)
    
    # Visualize with Grad-CAM for model interpretability
    print("\n--- Visualizing with Grad-CAM ---")
    visualize_with_grad_cam(model2, test_generator)

if __name__ == "__main__":
    main()
