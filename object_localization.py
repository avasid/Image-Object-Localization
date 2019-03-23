import os, sys
from keras.models import Model,load_model
from keras import backend as tf
sys.path.append(os.getcwd() + "/src")
from images2array import convert
from train import training
from pred_on_images import predictions

RESOLUTION = int(input("Enter resize resolution of images: "))


def run_all(RESOLUTION):
    convert(RESOLUTION)
    model = training(RESOLUTION)
    model.save('./model.h5')
    model = load_model('./model.h5')

# metric_function
def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*RESOLUTION
    x=tf.maximum(tf.minimum(x,RESOLUTION),0.0)
    y=predictions[:,1]*RESOLUTION
    y=tf.maximum(tf.minimum(y,RESOLUTION),0.0)
    width=predictions[:,2]*RESOLUTION
    width=tf.maximum(tf.minimum(width,RESOLUTION),0.0)
    height=predictions[:,3]*RESOLUTION
    height=tf.maximum(tf.minimum(height,RESOLUTION),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.tf.multiply(width,height)
    a2=tf.tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.tf.reduce_mean(sum)

# loss_function
def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*RESOLUTION)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.tf.where(condition,small_res,large_res)
    return tf.tf.reduce_mean(loss)

##################################

while True:
    print("1) Resize and convert images to array\n2) Train Model\n"
          "3) Load Model\n4) Test Model\n5) Run all\n6) Quit\n")
    choice = int(input("Choice: "))
    choice_dict = {1: convert,
                   2: training,
                   3: load_model,
                   4: predictions,
                   5: run_all,
                   6: exit}

    if choice == 2:
        model = training(RESOLUTION)
    elif choice == 3:
        name = input("Enter model name: ")
        print("Loading model..." + name)
        try:
            model = load_model(name, custom_objects1={'smooth_l1_loss': smooth_l1_loss, 'my_metric':my_metric})
        except:
            print("\nError in loading model\n")

    else:
        choice_dict[choice](RESOLUTION)


    print("Complete")
