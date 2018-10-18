#!/usr/bin/python
import tensorflow as tf
import os
from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
import time

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

tf.flags.DEFINE_boolean('distributed', False,
                        'Way of processing the code Distributed or Single Node')

tf.flags.DEFINE_string('node_root','','here is the folder for the node')

tf.flags.DEFINE_string('ps_hosts','','list of parameter servers')

tf.flags.DEFINE_string('worker_hosts','','list of worker servers')

tf.flags.DEFINE_integer('task_index','','server task index ')

tf.flags.DEFINE_string('job_name','','one of them : ps or worker')

tf.flags.DEFINE_string('input_path','','path of the input files on distributed training phase')

def main(argv):
    start_time = time.time()
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    config.distributed = FLAGS.distributed
    config.test_image_dir = os.path.join(FLAGS.node_root,'images')
    config.test_result_dir = os.path.join(FLAGS.node_root,'results')
    config.test_result_file = os.path.join(FLAGS.node_root,'results.cvs')
    config.replicas = len(FLAGS.worker_hosts.split(","))  
    config.task_index = FLAGS.task_index 
 
    if FLAGS.phase == 'train':
            # training phase

	   if FLAGS.distributed:
        	config.train_image_dir = FLAGS.input_path
                print config.train_image_dir

		ps_hosts = FLAGS.ps_hosts.split(",")
           	worker_hosts = FLAGS.worker_hosts.split(",")

            	# Create a cluster from the parameter server and worker hosts.
       	   	cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

            	# Create and start a server for the local task.
       	   	server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

		#with tf.device(tf.train.replica_device_setter(cluster=cluster)):
                #                global_step = tf.Variable(0)
				

		#with tf.device("/job:ps/task:0"):
    		#	global_step = tf.Variable(0, name="global_step")

	   	if FLAGS.job_name == "ps":
    			server.join()
  	   	elif FLAGS.job_name == "worker":
	   		with tf.device(tf.train.replica_device_setter(
        				worker_device="/job:worker/task:%d" % FLAGS.task_index,
        				cluster=cluster)):
				
				tf.reset_default_graph()
         			global_step = tf.contrib.framework.get_or_create_global_step()
		
                 #               global_step = tf.get_variable('global_step', [],
                 #                     initializer = tf.constant_initializer(0),
                 #                     trainable = False,
                 #                     dtype = tf.int32)
 
	  			data = prepare_train_data(config)
            			model = CaptionGenerator(config)
			
				init_op = tf.initialize_all_variables()
	
			is_chief = (FLAGS.task_index == 0)
    			# Create a "supervisor", which oversees the training process.
    			sv = tf.train.Supervisor(is_chief=is_chief,
                         	    logdir="/home/mauro.emc/image_captioning/tmp/logs",
                             	    init_op=init_op,
                             	    global_step=global_step,
                            	    save_model_secs=600)	
			
			with sv.prepare_or_wait_for_session(server.target) as sess:
				if is_chief:
            				sv.start_queue_runners(sess, [chief_queue_runner])
            				# Insert initial tokens to the queue.
            				sess.run(init_token_op)
            			sess.run(tf.global_variables_initializer())
				model.train(sess, data)
			sv.stop()
           else:
		with tf.Session() as sess:
			data = prepare_train_data(config)
		        model = CaptionGenerator(config)
            		sess.run(tf.global_variables_initializer())
            		if FLAGS.load:
                		model.load(sess, FLAGS.model_file)
            		if FLAGS.load_cnn:
                		model.load_cnn(sess, FLAGS.cnn_model_file)
            		tf.get_default_graph().finalize()
            		model.train(sess, data)
			

	
    elif FLAGS.phase == 'eval':
	 with tf.Session() as sess:
	 # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

    else:
         with tf.Session() as sess:
	    # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)
    print 'Total time in seconds :   ' + str(time.time() - start_time)
 

if __name__ == '__main__':
    tf.app.run()
