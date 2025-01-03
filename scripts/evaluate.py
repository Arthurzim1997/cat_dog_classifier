import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import re

# Diretório das imagens de teste (usando caminho relativo)
TEST_DIR = os.path.join(os.path.dirname(__file__), '../data/test')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.h5')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../results/predictions.txt')
ANNOTATIONS_FILE = os.path.join(os.path.dirname(__file__), '../data/test_annotations.txt')

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)

# Função para fazer a previsão em uma imagem
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)  # Mantendo a barra de progresso padrão do TensorFlow
    return 'dog' if prediction[0][0] > 0.5 else 'cat'

# Função para carregar anotações
def load_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split(',')
            annotations[img_name] = label.strip().lower()  # Certifique-se de que os labels estão em minúsculas e sem espaços extras
    return annotations

# Função para ordenar arquivos numericamente
def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

# Carregar anotações
annotations = load_annotations(ANNOTATIONS_FILE)

# Perguntar ao usuário se deseja fazer predições em todas as imagens ou em um número específico
use_all_images = input("Deseja fazer a predição em todas as imagens da pasta? (s/n): ").strip().lower()

# Listar as imagens de teste e ordenar numericamente
test_images = sorted(os.listdir(TEST_DIR), key=numerical_sort)

# Determinar quantas imagens processar
if use_all_images == 's':
    num_images = len(test_images)
else:
    num_images = int(input("Quantas imagens deseja usar para a predição?: ").strip())

# Perguntar ao usuário se deseja usar a verificação de porcentagem de acerto e erro
use_verification = input("Deseja usar a verificação de porcentagem de acerto e erro? (s/n): ").strip().lower()

# Inicializar contadores
dog_count = 0
cat_count = 0
correct_predictions = 0
na_count = 0

# Lista para armazenar resultados
results = []

# Fazer previsões nas imagens selecionadas
for i, img_name in enumerate(test_images[:num_images], start=1):
    img_path = os.path.join(TEST_DIR, img_name)
    result = predict_image(img_path)
    
    if use_verification == 's':
        # Verificar se a predição está correta
        correct_label = annotations.get(img_name, 'N/A')  # Usar 'N/A' se a anotação não existir
        if correct_label == 'N/A':
            is_correct = 'N/A'
            na_count += 1
        else:
            is_correct = (correct_label == result)
            if is_correct:
                correct_predictions += 1
        
        results.append(f'Imagem {i}/{num_images}: {img_name}, Predição: {result}, Correto: {is_correct}')
        print(f'Imagem {i}/{num_images}: {img_name}, Predição: {result}, Correto: {is_correct}')
    else:
        results.append(f'Imagem {i}/{num_images}: {img_name}, Predição: {result}')
        print(f'Imagem {i}/{num_images}: {img_name}, Predição: {result}')
    
    if result == 'dog':
        dog_count += 1
    else:
        cat_count += 1

# Calcular porcentagens
total_images = num_images
dog_percentage = (dog_count / total_images) * 100
cat_percentage = (cat_count / total_images) * 100

# Adicionar contagem e porcentagem ao resultado
results.append(f'\nTotal de imagens: {total_images}')
results.append(f'Total de Predição dog: {dog_count} ({dog_percentage:.2f}%)')
results.append(f'Total de Predição cat: {cat_count} ({cat_percentage:.2f}%)')

# Exibir contagem e porcentagem no terminal
print(f'\nTotal de imagens: {total_images}')
print(f'Total de Predição dog: {dog_count} ({dog_percentage:.2f}%)')
print(f'Total de Predição cat: {cat_count} ({cat_percentage:.2f}%)')

if use_verification == 's':
    # Calcular porcentagens de acerto, erro e N/A
    if total_images != na_count:
        accuracy_percentage = (correct_predictions / (total_images - na_count)) * 100
        error_percentage = 100 - accuracy_percentage
    else:
        accuracy_percentage = 0
        error_percentage = 0
    na_percentage = (na_count / total_images) * 100

    # Adicionar porcentagem de acerto, erro e N/A ao resultado
    results.append(f'Porcentagem de acerto sem N/A: {accuracy_percentage:.2f}%')
    results.append(f'Porcentagem de erro sem N/A: {error_percentage:.2f}%')
    results.append(f'Porcentagem de N/A: {na_percentage:.2f}%')

    # Exibir porcentagem de acerto, erro e N/A no terminal
    print(f'Porcentagem de acerto sem N/A: {accuracy_percentage:.2f}%')
    print(f'Porcentagem de erro sem N/A: {error_percentage:.2f}%')
    print(f'Porcentagem de N/A: {na_percentage:.2f}%')

# Salvar os resultados em um arquivo de texto
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    for line in results:
        f.write(line + '\n')

print(f'Resultados salvos em {OUTPUT_FILE}')