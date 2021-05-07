from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import random
import h5py
import os

os.environ['TFHUB_CACHE_DIR'] = 'D:\Development\TensorFlow\TFHUB_CACHE'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

batch_size = 16
epochs = 3
seed = 20

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def load_data():
    # Load Twitter BERT data
    data = h5py.File(os.curdir + '/data/training_data_BERT.h5', 'r')
    print('Loading train data...')
    input_ids_TRAIN = np.array(data['input_ids_vals_TRAIN'])
    mask_ids_TRAIN = np.array(data['input_mask_vals_TRAIN'])
    segment_ids_TRAIN = np.array(data['segment_ids_vals_TRAIN'])
    y_TRAIN = np.array(data['labels_TRAIN'])
    print('Loading test data...')
    input_ids_TEST = np.array(data['input_ids_vals_TEST'])
    mask_ids_TEST = np.array(data['input_mask_vals_TEST'])
    segment_ids_TEST = np.array(data['segment_ids_vals_TEST'])
    y_TEST = np.array(data['labels_TEST'])

    return input_ids_TRAIN, mask_ids_TRAIN, segment_ids_TRAIN, y_TRAIN,\
           input_ids_TEST, mask_ids_TEST, segment_ids_TEST, y_TEST


def build_bert_model():
    input1 = tf.keras.layers.Input(shape=(240,), batch_size=batch_size, dtype="int32", name='input_data')
    input2 = tf.keras.layers.Input(shape=(240,), batch_size=batch_size, dtype="int32", name='mask_data')
    input3 = tf.keras.layers.Input(shape=(240,), batch_size=batch_size, dtype="int32", name='segment_data')

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
    _, sequence_output = bert_layer([input1, input2, input3])
    bert_output = sequence_output

    flat1 = tf.keras.layers.Flatten()(bert_output)
    output = tf.keras.layers.Dense(2, activation='softmax')(flat1)
    
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=output)
    return model


def get_callbacks():
    # Prepare model saving directory and callback.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'TxtSentiment_model_BERT.{epoch:03d}.h5'
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, model_name)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True)

    callbacks = [checkpoint]
    return callbacks


def train_model(model, callbacks, input_ids_TRAIN, mask_ids_TRAIN, segment_ids_TRAIN, y_TRAIN,\
                input_ids_TEST, mask_ids_TEST, segment_ids_TEST, y_TEST):

    opt = tf.keras.optimizers.Adam
    model.compile(loss='categorical_crossentropy', optimizer=opt(learning_rate=2e-5), metrics=['accuracy'])
    print(model.summary())

    history = model.fit([input_ids_TRAIN, mask_ids_TRAIN, segment_ids_TRAIN], y_TRAIN, 
                         validation_data=([input_ids_TEST, mask_ids_TEST, segment_ids_TEST], y_TEST), 
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=callbacks)
    
    return history, epochs


#Visualise our training results
def training_results(history, EPOCHS):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy BERT')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')

    plt.title('Training and Validation Loss BERT')
    plt.savefig('./Bert_training_results.png')
    plt.show()


def main():
    print("Loading data......")
    input_ids_TRAIN, mask_ids_TRAIN, segment_ids_TRAIN, y_TRAIN,\
    input_ids_TEST, mask_ids_TEST, segment_ids_TEST, y_TEST = load_data()
    
    print("Getting model.....")
    model = build_bert_model()
    callbacks = get_callbacks()
    
    print("Start training....")
    history, epochs = train_model(model, callbacks, input_ids_TRAIN, mask_ids_TRAIN, segment_ids_TRAIN, y_TRAIN,\
                                  input_ids_TEST, mask_ids_TEST, segment_ids_TEST, y_TEST)

    print("Done!")
    training_results(history, epochs)


if __name__ == "__main__":
    main()