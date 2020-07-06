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

dataAugArgs = dict( rescale = 1./255,
                    rotation_range=0.2,
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

trainGen = trainGenerator(4,'datasets/SomiteTraceLibrary/input','train_frames','train_masks',dataAugArgs,save_to_dir = None)
valGen = trainGenerator(4,'datasets/SomiteTraceLibrary/input','val_frames','val_masks',dataAugArgs,save_to_dir = None)

weights_path = 'model_weights/'

m = model.unet()
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

m.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./logs/2log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'acc', verbose = 1,
                               min_delta = 0.01, patience = 3, mode = 'max')

tensorboard = TensorBoard(
    log_dir = './logs/bce/',
    write_graph = True,
    write_images = True
)

callbacks_list = [checkpoint, csv_logger, earlystopping, tensorboard]

results = m.fit_generator(trainGen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=valGen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
m.save('./model_weights/bce_Model.h5')

# train = trainGenerator(4, DATA_PATH, 'train_frames', 'train_masks', dataAugArgs, save_to_dir=None)
# model1 = unet()
# model_checkpoint = ModelCheckpoint('unetFirstRun.hdf5', monitor='loss', verbose = 1, save_best_only = True)
# model1.fit_generator(train,steps_per_epoch=10,epochs=1,callbacks=[model_checkpoint])