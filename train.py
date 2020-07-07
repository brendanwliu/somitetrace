# from model import 
import model
from somiteData import trainGenerator, testGenerator, saveResult
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.applications import VGG16
import os

dataAugArgs = dict( rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

DATA_PATH = 'datasets/SomiteTraceLibrary/input/'
FRAME_PATH = DATA_PATH + 'frames/'
MASK_PATH = DATA_PATH + 'masks/'

NO_OF_TRAINING_IMAGES = len(os.listdir(DATA_PATH + 'train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir(DATA_PATH + 'val_frames'))

NO_OF_EPOCHS = 35

BATCH_SIZE = 4

trainGen = trainGenerator(BATCH_SIZE ,'datasets/SomiteTraceLibrary/input','train_frames','train_masks', dataAugArgs, save_to_dir = None)
valGen = trainGenerator(BATCH_SIZE ,'datasets/SomiteTraceLibrary/input','val_frames','val_masks', dataAugArgs, save_to_dir = None)

weights_path = 'model_weights/'

m = model.unet()
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

m.compile(loss = model.dice_coef_loss,
              optimizer=opt,
              metrics=[model.dice_coef])

checkpoint = ModelCheckpoint(weights_path, monitor = model.dice_coef, 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./logs/1log.out', append=True, separator=';')

tensorboard = TensorBoard(
    log_dir = './logs/dice/',
    write_graph = True,
    write_images = True
)

callbacks_list = [checkpoint, csv_logger, tensorboard]

results = m.fit_generator(trainGen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=valGen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          shuffle=True,
                          callbacks=callbacks_list)
m.save('./model_weights/dice_Model.h5')