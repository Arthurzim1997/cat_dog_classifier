import os
import shutil

# Diretório raiz onde estão as imagens
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/train'))

# Diretórios para cada classe
CATS_DIR = os.path.join(TRAIN_DIR, 'cats')
DOGS_DIR = os.path.join(TRAIN_DIR, 'dogs')

# Criar diretórios de classes se não existirem
os.makedirs(CATS_DIR, exist_ok=True)
os.makedirs(DOGS_DIR, exist_ok=True)

# Mover imagens para os respectivos diretórios
for filename in os.listdir(TRAIN_DIR):
    # Ignorar os diretórios 'cats' e 'dogs'
    if filename in ['cats', 'dogs']:
        continue
    if filename.startswith('cat'):
        shutil.move(os.path.join(TRAIN_DIR, filename), os.path.join(CATS_DIR, filename))
    elif filename.startswith('dog'):
        shutil.move(os.path.join(TRAIN_DIR, filename), os.path.join(DOGS_DIR, filename))

print("Imagens reorganizadas com sucesso!")