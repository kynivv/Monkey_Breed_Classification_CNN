from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop


# Instantiating Mobilenet with Imagenet weights.
height = 224
width = 224

mobilenet = MobileNet(weights= 'imagenet', include_top= False, input_shape= (height, width, 3))

for layer in mobilenet.layers:
    layer.trainable = False

for i, layers in enumerate(mobilenet.layers) :
    print(f'Layer {i+1} -->', layers.__class__.__name__, layers.trainable)


# Creating a top model for the Monkey Breed Classification model.
def add_top_layers_to_MobileNet(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D() (top_model)
    top_model = Dense(1024, activation= 'relu') (top_model)
    top_model = Dense(1024, activation= 'relu') (top_model)
    top_model = Dense(512, activation= 'relu') (top_model)
    top_model = Dense(num_classes, activation= 'softmax') (top_model)
    return top_model


# Join the bottom and top models.
self_made_layers = add_top_layers_to_MobileNet(mobilenet, 10)
model = Model(inputs= mobilenet.input, outputs= self_made_layers)

model.compile(loss= 'categorical_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics= ['accuracy']
              )


# Instantiate ImageDataGenerators for augmenting the data.
train_dir = 'monkey_breed//training'
validation_dir = 'monkey_breed//validation'

train_gen = ImageDataGenerator(rescale= 1./255,
                               rotation_range= 45,
                               height_shift_range= 0.3,
                               width_shift_range= 0.3,
                               horizontal_flip= True,
                               fill_mode='nearest'
                               )

validation_gen = ImageDataGenerator(rescale= 1./255)


# Augmenting the data for Monkey Breed Classification.
training_data = train_gen.flow_from_directory(train_dir,
                                              batch_size= 1,
                                              shuffle= True,
                                              class_mode= 'categorical')
                                              
validation_data = validation_gen.flow_from_directory(validation_dir,
                                              batch_size= 1,
                                              shuffle= False,
                                              class_mode= 'categorical')


# Creating callbacks for the Monkey Breed Classification model.
checkpoint = ModelCheckpoint('monkey_breed_MobileNEt_V1.h5',
                             verbose= 1,
                             mode= 'min',
                             monitor= 'val_loss',
                             save_best_only= True
                             )

earlystopping = EarlyStopping(verbose= 1,
                              monitor= 'val_loss',
                              restore_best_weights= True,
                              patience= 3,
                              min_delta= 0)

callbacks = [earlystopping, checkpoint]


# Constants
train_samples = 1098
validation_samples = 272
epochs = 5
batch_size = 1


# Train the model
history = model.fit(training_data,
                    epochs= epochs,
                    steps_per_epoch= train_samples//batch_size,
                    validation_data= validation_data,
                    callbacks= callbacks,
                    validation_steps= validation_samples//batch_size)

