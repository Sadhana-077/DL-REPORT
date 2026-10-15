# ============================================================
# Chest X-Ray Pneumonia Classification with Data Augmentation
# SINGLE CELL - paste this whole thing in one Colab cell and Run
# No kaggle.json / no login needed - downloads dataset automatically
# ============================================================
!pip -q install kagglehub
import os, glob, random, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------- 1. AUTO DOWNLOAD DATASET (no login needed) ----------
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Dataset downloaded at:", path)

DATA_DIR = None
for root, dirs, _ in os.walk(path):
    if 'train' in dirs and 'test' in dirs:
        DATA_DIR = root; break
print("Using DATA_DIR:", DATA_DIR)

CLASSES = ['NORMAL', 'PNEUMONIA']
def get_files(split):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        for p in glob.glob(f'{DATA_DIR}/{split}/{c}/*'):
            if p.lower().endswith(('.jpeg', '.jpg', '.png')):
                paths.append(p); labels.append(i)
    return np.array(paths), np.array(labels)

full_paths, full_labels = get_files('train')
test_paths, test_labels = get_files('test')
print("Full train:", np.bincount(full_labels), "| Test:", np.bincount(test_labels))

# ---------- 2. SMALL DATASET (simulate limited medical data) ----------
N_PER_CLASS = 200
IMG_SIZE = (224, 224)
BATCH = 32

rng = np.random.RandomState(SEED)
idx = np.concatenate([rng.choice(np.where(full_labels == c)[0], N_PER_CLASS, replace=False) for c in [0, 1]])
small_paths, small_labels = full_paths[idx], full_labels[idx]
tr_p, va_p, tr_l, va_l = train_test_split(small_paths, small_labels, test_size=0.2,
                                          stratify=small_labels, random_state=SEED)
print("Train:", len(tr_p), "Val:", len(va_p), "Test:", len(test_paths))

# ---------- 3. DATA PIPELINE + AUGMENTATION ----------
AUTOTUNE = tf.data.AUTOTUNE
def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img.set_shape(IMG_SIZE + (3,))
    return img, tf.cast(label, tf.float32)

augmenter = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15, value_range=(0, 255)),
])

def make_ds(paths, labels, training=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training: ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_img, num_parallel_calls=AUTOTUNE).batch(BATCH)
    if augment:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0, 255), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_plain = make_ds(tr_p, tr_l, training=True, augment=False)
train_aug   = make_ds(tr_p, tr_l, training=True, augment=True)
val_ds      = make_ds(va_p, va_l)
test_ds     = make_ds(test_paths, test_labels)

# ---------- 4. MODEL ----------
def build_model():
    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input(IMG_SIZE + (3,))
    x = layers.Rescaling(1/127.5, offset=-1)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inp, out), base

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=tr_l)
class_weight = {0: cw[0], 1: cw[1]}
metrics = ['accuracy', keras.metrics.AUC(name='auc')]

def train(train_ds, head_epochs=5, ft_epochs=3):
    model, base = build_model()
    cb = [keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)]
    model.compile(keras.optimizers.Adam(1e-3), 'binary_crossentropy', metrics=metrics)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=class_weight, callbacks=cb, verbose=2)
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    model.compile(keras.optimizers.Adam(1e-5), 'binary_crossentropy', metrics=metrics)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                   class_weight=class_weight, callbacks=cb, verbose=2)
    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    return model, hist

print("\n===== TRAINING WITHOUT AUGMENTATION =====")
model_plain, hist_plain = train(train_plain)

print("\n===== TRAINING WITH AUGMENTATION =====")
model_aug, hist_aug = train(train_aug)

# ---------- 5. EVALUATE ----------
def evaluate(model, name):
    probs = model.predict(test_ds, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    print(f"\n===== {name} : TEST RESULTS =====")
    print(classification_report(test_labels, preds, target_names=CLASSES, digits=3))
    acc = (preds == test_labels).mean()
    return preds, acc, confusion_matrix(test_labels, preds)

preds_plain, acc_plain, cm_plain = evaluate(model_plain, "WITHOUT augmentation")
preds_aug,   acc_aug,   cm_aug   = evaluate(model_aug,   "WITH augmentation")

print(f"\nFINAL COMPARISON -> No-aug test accuracy: {acc_plain:.3f} | Aug test accuracy: {acc_aug:.3f}")

# ---------- 6. PLOTS ----------
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].plot(hist_plain['val_accuracy'], label='No-aug'); ax[0].plot(hist_aug['val_accuracy'], label='Aug')
ax[0].set_title('Validation Accuracy'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hist_plain['val_loss'], label='No-aug'); ax[1].plot(hist_aug['val_loss'], label='Aug')
ax[1].set_title('Validation Loss'); ax[1].legend(); ax[1].grid(alpha=.3)
for a, cm, title in zip(ax[2:], [cm_plain, cm_aug], ['No-aug CM', 'Aug CM']):
    a.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
    a.set_xticks([0, 1]); a.set_xticklabels(CLASSES); a.set_yticks([0, 1]); a.set_yticklabels(CLASSES)
    a.set_title(title)
plt.tight_layout(); plt.show()
