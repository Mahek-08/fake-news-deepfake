# deepfake/train_deepfake.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Directory structure:
# data/deepfake/train/real/..., data/deepfake/train/fake/...
# data/deepfake/val/real/..., data/deepfake/val/fake/...

train_dir = 'data/deepfake/train'
val_dir = 'data/deepfake/val'
img_size = (224, 224)
batch_size = 16

train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_flow = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

base = Xception(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base.input, outputs=output)

# Freeze base
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(train_flow, validation_data=val_flow, epochs=8)

# Unfreeze some layers and fine-tune
for layer in base.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=6)

# Save
os.makedirs('deepfake/models', exist_ok=True)
model.save('deepfake/models/deepfake_xception.h5')
print('Saved deepfake model')
