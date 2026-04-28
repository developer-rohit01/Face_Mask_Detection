from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to dataset
data_dir = "dataset/"

# Create generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("Classes:", train_data.class_indices)
print("Training samples:", train_data.samples)
print("Validation samples:", val_data.samples)