
Use of AI
---------
Course: COE 379L — Project 3 (Damage vs. No-Damage Classification)
Student: Mahin Naveen
Date: 2025-11-19

This document lists instances where I used ChatGPT. Each entry includes: **Tool**, **Prompt**, **Output**, and a **small code example** I wrote based on that guidance.  
In my notebook, I reference entries with inline comments like: `# See Use of AI [N]`.

[1]. Tool: ChatGPT
Prompt: How should I handle a ~2:1 class imbalance with tf.data? Class weights vs. oversampling?
Output:
  Use **class weights** computed from training counts; keep augmentations modest; avoid any oversampling on val/test; only consider train‑only oversampling if minority recall remains weak.
Code example (weights used in Model.fit):
```python
# See Use of AI [1]
from collections import Counter

train_counts = Counter(train_labels)  # e.g., {'damage': 9918, 'no_damage': 5007}
N = sum(train_counts.values())
class_weight = {
    1: N / (2.0 * train_counts['damage']),     # label 1 for 'damage'
    0: N / (2.0 * train_counts['no_damage'])   # label 0 for 'no_damage'
}
hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, class_weight=class_weight, callbacks=cb)
```

[2]. Tool: ChatGPT
Prompt: Why does the flattened dense model perform poorly, even with class weights?
Output:
  Flattening removes spatial structure; CNNs capture locality/hierarchy. Keep the dense model only as a **baseline** to quantify the value of convolution.
Code example (baseline annotation in build_dense_ann):
```python
# See Use of AI [2]
inputs = keras.Input((64, 64, 1))
x = layers.Flatten()(inputs)  # spatial info lost: baseline only
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
```

[3]. Tool: ChatGPT
Prompt: LeNet‑5 at 32×32 gray plateaus. Should I increase resolution and keep RGB for satellite tiles?
Output:
  Yes. Increase input to ~150×150 and keep RGB; use a deeper CNN to capture subtle roof/debris cues.
Code example (Alternate‑LeNet input & first block):
```python
# See Use of AI [3]
ALT_IMG_SIZE = (150, 150)
inputs = keras.Input((*ALT_IMG_SIZE, 3))  # RGB retained
x = layers.Conv2D(32, 3, activation='relu', padding='valid')(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='valid')(x)
x = layers.MaxPool2D()(x)
```

[4]. Tool: ChatGPT
Prompt: Validation metrics bounce after ~8 epochs. Which callbacks/LR schedule should I use?
Output:
  Lower base LR (≈3e‑4), add **EarlyStopping**, **ReduceLROnPlateau**, and **ModelCheckpoint** saving best weights only.
Code example (callbacks block used in all models):
```python
# See Use of AI [4]
opt = keras.optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                                                                 keras.metrics.AUC(name='auc')])
cb = [
    keras.callbacks.EarlyStopping(patience=4, monitor='val_auc', mode='max', restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_auc', mode='max'),
]
```

[5]. Tool: ChatGPT
Prompt: I’m hitting shape/dtype issues with `tf.io.decode_image` and RGB/gray toggling. What’s the robust pattern?
Output:
  Decode bytes → set channels (1 or 3) → resize explicitly → convert to float32 in [0,1]; keep map deterministic; use AUTOTUNE.
Code example (shared loader):
```python
# See Use of AI [5]
def load_image(path, to_gray=False, size=(150, 150)):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=(1 if to_gray else 3), expand_animations=False)
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img
```

[6]. Tool: ChatGPT
Prompt: Which metric should I use to pick the best model on an imbalanced dataset?
Output:
  Choose by **validation F1** (or macro‑F1); report AUC as a complement. Evaluate the chosen model **once** on the test set.
Code example (selection logic):
```python
# See Use of AI [6]
results = {'DenseANN': {'val_f1': val_f1_dense},
           'LeNet5':   {'val_f1': val_f1_lenet},
           'AltLeNet': {'val_f1': val_f1_alt}}
best_name = max(results, key=lambda k: results[k]['val_f1'])
```

[7]. Tool: ChatGPT
Prompt: What should my evaluation cell include for clarity with TAs?
Output:
  Print **classification_report**, **confusion_matrix**, and **AUC**, alongside accuracy; include a short note on false negatives.
Code example (evaluation cell):
```python
# See Use of AI [7]
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

probs = model.predict(ds_test).ravel()
preds = (probs > 0.5).astype('int32')
print(classification_report(y_test, preds, target_names=['no_damage','damage']))
print(confusion_matrix(y_test, preds))
print('AUC:', roc_auc_score(y_test, probs))
```

[8]. Tool: ChatGPT
Prompt: With Part 3 (Docker/server) exempt, which artifacts should I still save for future deployment?
Output:
  Save `best_model.keras`, `summary.json` (metrics + metadata), and `label_map.json` consistent with preprocessing.
Code example (artifact export):
```python
# See Use of AI [8]
model.save('best_model.keras')
summary = {'val_f1': float(val_f1_alt), 'test_f1': float(test_f1), 'input_size': [150,150,3]}
with open('summary.json', 'w') as f: json.dump(summary, f)
with open('label_map.json', 'w') as f: json.dump({0:'no_damage', 1:'damage'}, f)
```

[9]. Tool: ChatGPT
Prompt: Help me structure my Part 4 report and tell me where to change tone to keep it more professional without changing technical content.
Output:
  Outline: Data Prep → Models (with motivation) → Evaluation → Deployment Note → Conclusion; keep numbers precise and concise.
Code usage:
```text
Narrative guidance only; I wrote, verified, and formatted all content myself.
```

[10]. Tool: ChatGPT
Prompt: Draft a brief AI‑use disclosure for the top of my notebook and the report.
Output:
  Two–three sentences noting AI assistance for concept discussion, debugging, and editorial polish; code is mine; usage complies with course policy.
Code example (markdown blurb in the first cell of the notebook):
```markdown
**Use of AI.** I used ChatGPT to discuss design choices, interpret errors, and polish wording.
All code was written and run by me. I verified any suggestions and made final decisions myself,
consistent with the course AI policy.
```

-- End of document --
