from model import unet
from somiteData import trainGenerator, testGenerator, saveResult
from keras.callbacks import ModelCheckpoint

dataAugArgs = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train = trainGenerator(1, 'datasets/SomiteTraceLibrary/input/train', 'image', 'label', dataAugArgs, save_to_dir=None)
model1 = unet()
model_checkpoint = ModelCheckpoint('unetFirstRun.hdf5', monitor='loss', verbose = 1, save_best_only = True)
model1.fit_generator(train,steps_per_epoch=10,epochs=1,callbacks=[model_checkpoint])

test = testGenerator("datasets/SomiteTraceLibrary/input/test/image",target_size = (128,128))
results = model1.predict_generator(test,180, verbose=1)
saveResult("datasets/SomiteTraceLibrary/input/test/results", results)