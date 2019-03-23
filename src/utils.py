import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import random


def getdata():
    # read data and shuffle

    f=open("./id_to_data","rb+")
    data=pickle.load(f)
    f.close()

    # Shuffle
    len_data = len(data)
    index=[i for i in range(len_data)]
    random.shuffle(index)

    data=data[index]
    split_boundary = int(len_data*0.75) # 0.75 split ratio
    data_train=data[0:split_boundary]
    data_test=data[split_boundary:]

    f=open("./id_to_box","rb+")
    box=pickle.load(f)
    f.close()

    box=box[index]
    box_train=box[0:split_boundary]
    box_test=box[split_boundary:]

    f = open("./id_to_mean", "rb+")
    mean = pickle.load(f)
    f.close()

    mean = mean[index]
    # mean_train = mean[0:split_boundary]
    mean_test = mean[split_boundary:]

    f = open("./id_to_std", "rb+")
    std = pickle.load(f)
    f.close()

    std = std[index]
    # std_train = std[0:split_boundary]
    std_test = std[split_boundary:]

    with open("./id_to_data_test", "wb+") as fh:
        pickle.dump(data_test,fh ,protocol=4)

    with open("./id_to_box_test", "wb+") as fh:
        pickle.dump(box_test,fh ,protocol=4)

    with open("./id_to_mean_test", "wb+") as fh:
        pickle.dump(mean_test,fh ,protocol=4)

    with open("./id_to_std_test", "wb+") as fh:
        pickle.dump(std_test,fh ,protocol=4)

    return data_train,box_train,data_test,box_test


def plot_model(model_details):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_details.history['my_metric'])+1),model_details.history['my_metric'])
    axs[0].plot(range(1,len(model_details.history['val_my_metric'])+1),[1.7*x for x in model_details.history['val_my_metric']])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['my_metric'])+1),len(model_details.history['my_metric'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig("model.png")

