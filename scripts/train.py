import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configurações
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/train'))

# Verifique se o diretório existe
if not os.path.exists(TRAIN_DIR):
    raise ValueError(f"O diretório especificado não existe: {TRAIN_DIR}")

# Listar os arquivos no diretório para verificação
print(f"Arquivos no diretório {TRAIN_DIR}:")
print(os.listdir(TRAIN_DIR))

# Geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=123
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True,
    seed=123
)

# Verificar se os diretórios estão corretos
print("Classes encontradas: ", train_generator.class_indices)
print("Total de imagens de treino: ", train_generator.samples)
print("Total de imagens de validação: ", validation_generator.samples)

# Construção do modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Salvando o modelo
if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))):
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
model.save(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/model.h5')))

print("Modelo treinado e salvo com sucesso!")