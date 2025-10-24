from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D

def ASL_model(input_shape=(30, 225), num_classes=10):
    model = Sequential()

    # Khối CNN
    model.add(Conv1D(128, kernel_size=3, activation='relu' , input_shape=input_shape))  
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Dropout(0.3))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))  
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Dropout(0.3))
    

    # Khối LSTM
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(GRU(128)) 
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Khối Dense
    model.add(Dense(64, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))  

    model.summary()

    return model
