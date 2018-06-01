import os, sys

import numpy as np
import tensorflow as tf
import datetime
#from dgmodel import LeNetModel
from mstnmodel import LeNetModel
#from svhnmodel import SVHNModel
#from drcnmodel import DRCNModel
#sys.path.insert(0, '../../utils')
from mnist import MNIST
from svhn import SVHN
#from usps import USPS
from preprocessing import preprocessing

import math
from tensorflow.contrib.tensorboard.plugins import projector

tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 100000, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '256,257', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')

NUM_CLASSES = 10

TRAIN_FILE='svhn'
TEST_FILE='mnist'
print TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE
print TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE
print TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE



TRAIN=SVHN('data/svhn',split='train',shuffle=True)
VALID=MNIST('data/mnist',split='test',shuffle=True)
TEST=MNIST('data/mnist',split='test',shuffle=False)


FLAGS = tf.app.flags.FLAGS
MAX_STEP=10000

def decay(start_rate,epoch,num_epochs):
        return start_rate/pow(1+0.001*epoch,0.75)

def adaptation_factor(x):
	#return 1.0
	#return 0.25
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return min(lamb,1.0)
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
    
    adlamb=tf.placeholder(tf.float32,name='adlamb')
    num_update=tf.placeholder(tf.float32,name='num_update')
    decay_learning_rate=tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)
    is_training=tf.placeholder(tf.bool)    
    time=tf.placeholder(tf.float32,[1])

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = LeNetModel(num_classes=NUM_CLASSES, image_size=28,is_training=is_training,dropout_keep_prob=dropout_keep_prob)
    # Placeholders
    x_s = tf.placeholder(tf.float32, [None, 32, 32, 3],name='x')
    x_t = tf.placeholder(tf.float32, [None, 28, 28, 1],name='xt')
    x=preprocessing(x_s,model)
    xt=preprocessing(x_t,model)
    tf.summary.image('Source Images',x)
    tf.summary.image('Target Images',xt)
    print 'x_s ',x_s.get_shape()
    print 'x ',x.get_shape()
    print 'x_t ',x_t.get_shape()
    print 'xt ',xt.get_shape()
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES],name='y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES],name='yt')
    loss = model.loss(x, y)
    # Training accuracy of the model
    source_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    source_correct=tf.reduce_sum(tf.cast(source_correct_pred,tf.float32))
    source_accuracy = tf.reduce_mean(tf.cast(source_correct_pred, tf.float32))
    
    G_loss,D_loss,sc,tc=model.adloss(x,xt,y,yt)
    
    # Testing accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    correct=tf.reduce_sum(tf.cast(correct_pred,tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    update_op = model.optimize(decay_learning_rate,train_layers,adlamb,sc,tc)
	
    D_op=model.adoptimize(decay_learning_rate,train_layers)
    optimizer=tf.group(update_op,D_op)
    
    train_writer=tf.summary.FileWriter('./log/tensorboard')
    train_writer.add_graph(tf.get_default_graph())
    config=projector.ProjectorConfig()
    embedding=config.embeddings.add()
    embedding.tensor_name=model.feature.name
    embedding.metadata_path='domain.csv'
    projector.visualize_embeddings(train_writer,config)
    tf.summary.scalar('G_loss',model.G_loss)
    tf.summary.scalar('D_loss',model.D_loss)
    tf.summary.scalar('C_loss',model.loss)
    tf.summary.scalar('SA_loss',model.Semanticloss)
    tf.summary.scalar('Training Accuracy',source_accuracy)
    tf.summary.scalar('Testing Accuracy',accuracy)
    merged=tf.summary.merge_all()




    print '============================GLOBAL TRAINABLE VARIABLES ============================'
    print tf.trainable_variables()
    #print '============================GLOBAL VARIABLES ======================================'
    #print tf.global_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
	saver=tf.train.Saver()
	#saver.restore(sess,'log/checkpoint')
        # Load the pretrained weights
        #model.load_original_weights(sess, skip_layers=train_layers)
	train_writer.add_graph(sess.graph)
        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        #print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))
	gd=0
        step = 1
	for epoch in range(300000):
            # Start training
	    gd+=1
	    lamb=adaptation_factor(gd*1.0/MAX_STEP)
	    #rate=decay(FLAGS.learning_rate,gd,MAX_STEP)
	    power=gd/10000	    
	    rate=FLAGS.learning_rate
	    tt=pow(0.1,power)
	    batch_xs, batch_ys = TRAIN.next_batch(FLAGS.batch_size)
            Tbatch_xs, Tbatch_ys = VALID.next_batch(FLAGS.batch_size)
	    #print batch_xs.shape
            #print Tbatch_xs.shape
            summary,_,closs,gloss,dloss,smloss=sess.run([merged,optimizer,model.loss,model.G_loss,model.D_loss,model.Semanticloss], feed_dict={x_s: batch_xs,x_t: Tbatch_xs,time:[1.0*gd],decay_learning_rate:rate,adlamb:lamb,is_training:True,y: batch_ys,dropout_keep_prob:0.5,yt:Tbatch_ys})
	    train_writer.add_summary(summary,gd)
	
            step += 1
            if gd%250==0:
		epoch=gd/(72357/100)
	        print 'lambda: ',lamb
	        print 'rate: ',rate
		print 'Epoch {5:<10} Step {3:<10} C_loss {0:<10} G_loss {1:<10} D_loss {2:<10} Sem_loss {4:<10}'.format(closs,gloss,dloss,gd,smloss,epoch)
                print("{} Start validation".format(datetime.datetime.now()))
                test_acc = 0.
                test_count = 0
		print 'test_iter ',len(TEST.labels)
                for _ in xrange((len(TEST.labels))/5000):
                    batch_tx, batch_ty = TEST.next_batch(5000)
		    #print TEST.pointer,'   ',TEST.shuffle
                    acc = sess.run(correct, feed_dict={x_t: batch_tx, yt: batch_ty, is_training:True,dropout_keep_prob: 1.})
                    test_acc += acc
                    test_count += 5000
                print test_acc,test_count
                test_acc /= test_count
		if epoch==300:
		    return
		    
                #batch_tx, batch_ty = TEST.next_batch(len(TEST.labels))
		#test_acc=sess.run(accuracy,feed_dict={x_t:batch_tx,y:batch_ty,is_training:False,dropout_keep_prob:1.0})
		print len(batch_tx)
	        print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

		if gd%10000==0 and gd>0:
		    #saver.save(sess,'./log/mstn2model'+str(gd)+'.ckpt')
		    #print 'tensorboard --logdir ./log/tensorboard'
		    #return
		    pass 
                #print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            #save checkpoint of the model
            #checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            #save_path = saver.save(sess, checkpoint_path)

            #print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

if __name__ == '__main__':
    tf.app.run()
