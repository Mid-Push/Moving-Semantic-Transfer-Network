import os, sys
import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
import datetime
from mstnmodel import AlexNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor
import math
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '256,257', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')

NUM_CLASSES = 31
TRAINING_FILE = '../data/amazon_list.txt'
VAL_FILE = '../data/webcam_list.txt'
FLAGS = tf.app.flags.FLAGS
MAX_STEP=10000

def decay(start_rate,epoch,num_epochs):
        return start_rate/pow(1+0.001*epoch,0.75)

def adaptation_factor(x):
	if x>=1.0:
		return 1.0
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return lamb
def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()
    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3],'x')
    xt = tf.placeholder(tf.float32, [None, 227, 227, 3],'xt')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES],'y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES],'yt')
    adlamb=tf.placeholder(tf.float32)
    decay_learning_rate=tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel(num_classes=NUM_CLASSES, dropout_keep_prob=dropout_keep_prob)
    loss = model.loss(x, y)
    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct=tf.reduce_sum(tf.cast(correct_pred,tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #G_loss,D_loss=model.wganloss(x,xt,FLAGS.batch_size,10.0)
    G_loss,D_loss,sc,tc=model.adloss(x,xt,y,10)
    target_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    target_correct=tf.reduce_sum(tf.cast(target_correct_pred,tf.float32))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_pred, tf.float32))
    train_op = model.optimize(decay_learning_rate, train_layers,adlamb,sc,tc)
	
    D_op=model.adoptimize(decay_learning_rate,train_layers)
    optimizer=tf.group(train_op,D_op)
    
    train_writer=tf.summary.FileWriter('./log/tensorboard_restore')
    train_writer.add_graph(tf.get_default_graph())
    tf.summary.scalar('Testing Accuracy',target_accuracy)
    merged=tf.summary.merge_all()

    print '============================GLOBAL TRAINABLE VARIABLES ============================'
    print tf.trainable_variables()
    #print '============================GLOBAL VARIABLES ======================================'
    #print tf.global_variables()
    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2: 
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None
    print '==================== MULTI SCALE==================================================='
    print multi_scale
    train_preprocessor = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                           output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    Ttrain_preprocessor = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                           output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES, output_size=[227, 227],multi_scale=multi_scale,istraining=False)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    Ttrain_batches_per_epoch = np.floor(len(Ttrain_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
	saver=tf.train.Saver()
	train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        #model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, "./trained_mstn_model/10000.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))
        gs=0
	gd=0
	for epoch in range(FLAGS.num_epochs):
            #print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1
            # Start training
            while step < train_batches_per_epoch:
		gd+=1
	    	lamb=adaptation_factor(gd*1.0/MAX_STEP)
	    	rate=decay(FLAGS.learning_rate,gd,MAX_STEP)
                if gd%1==0:
                    print("{} Start validation".format(datetime.datetime.now()))
                    test_acc = 0.
                    test_count = 0

                    for _ in range((len(val_preprocessor.labels))):
                        batch_tx, batch_ty = val_preprocessor.next_batch(1)
                        acc = sess.run(correct, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
                        test_acc += acc
                        test_count += 1
                    print test_acc,test_count
                    test_acc /= test_count
	            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
                    # Reset the dataset pointers
                    val_preprocessor.reset_pointer()
		
                    
		    return
		 	

if __name__ == '__main__':
    tf.app.run()
