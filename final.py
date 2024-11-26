"""
IMPORT STATEMENTS
"""
# import statements
import pandas as pd
import numpy as np
import string

# model imports - sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# sequential imports - tensor flow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

"""
FUNCTIONS FOR PRE-PROCESSING
"""
# FUNCTION: clean document
def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return ' '.join(tokens)

# Load the dataset
df = pd.read_csv('Emotion_classify_Data.csv')
# clean document
df['cleaned_text'] = df['Comment'].apply(clean_doc)

# Tokenize the text
# "<00V> for unknown text"
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# fit tokenizer on the text
tokenizer.fit_on_texts(df['cleaned_text'])
# convert text to sequence
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
# pad the sequence to length of 50 words
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post')

# encode the labels
label_encoder = LabelEncoder()
# encode the emotions (i.e. anger, fear, joy) to a label
encoded_labels = label_encoder.fit_transform(df['Emotion'])

# calculate class weights for potential class imbalances
class_weights = compute_class_weight('balanced', np.unique(encoded_labels), encoded_labels)
class_weight_dict = dict(enumerate(class_weights))

# train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# get the number of unique words (vocab)
vocab_size = len(tokenizer.word_index) + 1
# max seq length
seq_length = 50  
# get number of classes for generalization
num_classes = len(label_encoder.classes_)

"""
LSTM Model
"""
# initialize lstm model
inputs = Input(shape=(seq_length,))
# custom embedding 
x = Embedding(input_dim=vocab_size, output_dim=128, input_length=seq_length)(inputs)
# LSTM layer taking in 128 units/neurons
x = LSTM(128, return_sequences=False)(x)
# drop out layer for regularization
x = Dropout(0.2)(x)
# softmax activation
outputs = Dense(num_classes, activation='softmax')(x)

# build model
model = Model(inputs=inputs, outputs=outputs)
# compile model with "adam" optimizer, "sparse_categorical_crossentropy" for mutually exclusive classes, "accuracy"
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# get model summary
model.summary()

# callback - save best version of the model during training
checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
# learning rate scheduler - reduces learning rate if performances plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)  

# train the model
# train for 30 epochs with a batch size of 32
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), 
                    class_weight=class_weight_dict, callbacks=[checkpoint, lr_scheduler])

# evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# calculate & output classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# calculate confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

"""
FUNCTION FOR PREDICTION & EVALUATING
"""
# FUNCTION: predict function
def classify_text(model, tokenizer, label_encoder, text, seq_length):
    # clean the text
    cleaned_text = clean_doc(text)
    # get sequences
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    # pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=seq_length, padding='post')
    # get prediction
    prediction = model.predict(padded_sequence)
    # get prediction label
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# sample text
example_text = "I seriously hate one subject to death but now I feel reluctant to drop it"
# predict emotion for this text
print("Predicted Emotion:", classify_text(model, tokenizer, label_encoder, example_text, seq_length))