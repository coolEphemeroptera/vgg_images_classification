from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from keras.models import Model
import utils

# base 模块
def bn(x):
    return BatchNormalization(axis=-1)(x)

def conv2d(x,n_fmaps,k=3):
    return Conv2D(n_fmaps, (k, k), strides=(1,1), activation='relu', padding='same')(x)

def maxpool(x):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

def dense(x,n_units,activation="relu"):
    x = Dropout(0.2)(x)
    return Dense(n_units,activation=activation)(x)

def softmax(x,n_classes):
    return Dense(n_classes,activation='softmax',name='the_labels')(x)

# vgg_16
def vgg_16(inputs, n_classes):
    # cnn-1
    x = bn(conv2d(inputs,32))
    x = bn(conv2d(x, 32))
    x = maxpool(x)

    # cnn-2
    x = bn(conv2d(x, 64))
    x = bn(conv2d(x, 64))
    x = maxpool(x)

    # cnn-3
    x = bn(conv2d(x, 128))
    x = bn(conv2d(x, 128))
    x = bn(conv2d(x, 128, k=1))
    x = maxpool(x)

    # cnn-4
    x = bn(conv2d(x, 256))
    x = bn(conv2d(x, 256))
    x = bn(conv2d(x, 256, k=1))
    x = maxpool(x)

    # cnn-5
    x = bn(conv2d(x, 256))
    x = bn(conv2d(x, 256))
    x = bn(conv2d(x, 256, k=1))
    x = maxpool(x)

    # fc-6,7,8
    x = Reshape([7*7*256])(x)
    x = dense(x, 2048)
    x = dense(x, 2048)
    x = dense(x, 512)

    # softmax
    x = softmax(x,n_classes)
    return x


if __name__ == "__main__":
    # 超参数
    IMG_SIZE = (224, 224)
    N_CLASSES = 105
    BATCH_SIZE = 32
    EPOCHS = 20
    train_file = "./data.train"
    dev_file = "./data.dev"
    test_file = "./data.test"

    # 数据准备
    train_lst = utils.reading(train_file)
    dev_lst = utils.reading(dev_file)
    test_lst = utils.reading(test_file)

    dev_data,dev_labels = utils.loading(dev_lst,n_classes=N_CLASSES)
    test_data, test_labels = utils.loading(test_lst, n_classes=N_CLASSES)
    train_generator = utils.data_generator(train_lst,batch_size=BATCH_SIZE,n_classes=N_CLASSES)

    # 模型 输入/输出
    inputs = Input(name='the_inputs', shape=(IMG_SIZE[0], IMG_SIZE[1],3), dtype='float32')

    # 编译模型
    model = Model(inputs=inputs, outputs=vgg_16(inputs, N_CLASSES))
    model.summary()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    # 训练
    n_batchs = len(train_lst)//BATCH_SIZE
    model.fit_generator(generator=train_generator,steps_per_epoch=n_batchs,epochs=EPOCHS,
                        validation_data=(dev_data,dev_labels),validation_steps=n_batchs)

