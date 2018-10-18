#!/home/mauro.emc/miniconda2/bin/python

import os
import shutil as sh
import sys
import numpy as np
import subprocess
import time
from tensorflow_on_slurm import tf_config_from_slurm
from config import Config

class DataPreparation():
	def __init__(self,ps_nodes,curpath,dirtest,dirtrain,type_mode,python_dir):

		self.curpath = curpath
		self.data_prepared = False
		if type_mode == 'train':
			self.cluster, self.index_task_ps, self.index_task_workers  = tf_config_from_slurm(ps_nodes)
	        self.homepath = ''
	        self.time_format = "%a, %d %b %Y %H:%M:%S"
		self.dirtest = dirtest
		self.dirtrain = dirtrain
		self.python_dir = python_dir
		self.list_dirs = []

	def create_dir(self,num_nodes,type_mode,root_name):
	        
		curpath = self.curpath
                root_dest = os.path.join(curpath,root_name) 
		if type_mode == 'train':
			dir_source=self.dirtrain
	                print "train:" + str(dir_source)

		else:
			dir_source=self.dirtest
	                print "inference:" + str(dir_source)

                lst_files=os.listdir(dir_source)

		#create dictionary with the files in each node
		print self.data_prepared
		print dir_source
		if not self.data_prepared:
			dir_files = {}
			print ' Delete tree: ' + root_dest
                        sh.rmtree(root_dest)
                        print ' Create root: ' + root_dest
                        os.mkdir(root_dest)
			if type_mode == 'train':
			     #Get the last task it means get the total number of servers. Add one beacuse tasks start with 0
			     num_nodes=self.index_task_workers[-1] + 1
 			     num_files_p_dir = np.int(np.floor(len(lst_files)*1.0/num_nodes)) 
			     print 'Number of nodes to split: ' + str(num_nodes)
			     print 'Number of files per node: ' + str(num_files_p_dir) 
			     for i in range(0,num_nodes):
				server = self.cluster['worker'][i][0:6]
				start = i * num_files_p_dir
                                stop = start + num_files_p_dir
				# Select the files for the folder
                                dir_files[i] = [lst_files[x] for x in range(start,stop)]
				print len(dir_files[i])
                                path_node_results = os.path.join(root_dest,str(i),'results') 
				path_node = os.path.join(root_dest,str(i),'images')
				# Create the images and results folder
				bashCommand = 'ssh ' + server + ' "mkdir -p ' + path_node + ';mkdir ' + path_node_results + '"' 
				exit_status = os.system(bashCommand)
				print time.strftime(self.time_format,time.gmtime()) + ' ' + ' Server/Node ' + str(server) + '/' + str(i) + ' : Exit Status (Zero is ok) : ' + str(exit_status)
				
				# Create batch process to copy the files in the folders
				batch = 10
				total_n_batches = np.int(np.ceil(len(dir_files[i])*1.0/batch))
				for j in range(0,total_n_batches):
					b_start = j * batch  
					b_stop = b_start + batch
					batch_files = ' '.join(dir_files[i][b_start:b_stop])
					bashCommand = 'ssh ' + server + ' "cd ' + dir_source + ';nohup cp ' + batch_files + ' ' + path_node + '  < /dev/null &"' 
					exit_status = os.system(bashCommand)
                                	print time.strftime(self.time_format,time.gmtime()) + ' Server/Node ' + server + '/' + str(i) + ' : Copying Files  -  ' + '{0:.2f}%'.format((j*1.)/total_n_batches*100) + ' : Exit Status (Zero is ok) : ' + str(exit_status)

				print time.strftime(self.time_format,time.gmtime()) + ' Server/Node ' + server + '/' + str(i) + ' : Total of files transfered : ' + str(len(dir_files[i]))
				self.data_prepared = True
			else:
			     num_files_p_dir = np.int(np.floor(len(lst_files)/num_nodes))
			     for i in range(0,num_nodes):
				start = i * num_files_p_dir 
				stop = start + num_files_p_dir 
				print start,stop
				dir_files[i] = [lst_files[x] for x in range(start,stop)]
				os.mkdir(os.path.join(root_dest,str(i)))
				os.mkdir(os.path.join(root_dest,str(i),'results'))
				print time.strftime(self.time_format,time.gmtime()) + ' ' + hostname + ' Directory tree creation for node :' + str(i)

			     for i in range(0,num_nodes):
				lst_files_node = dir_files[i]
				path_node = os.path.join(root_dest,str(i),'images')
				os.mkdir(path_node)
				count_files = 0 
				for j in range(0,len(lst_files_node)):
					if not self.data_prepared:
						path_source = os.path.join(dir_source,lst_files_node[j])	
						path_dest = os.path.join(path_node,lst_files_node[j])
						sh.copy2(path_source,path_dest)
						count_files +=1
				print time.strftime(self.time_format,time.gmtime()) + ' ' + hostname + ' Images saved for node :' + str(i) + ' files saved : ' + str(count_files)
		return		


	def create_jobs_inference(self,num_nodes,root_name):
		curpath = self.curpath
		num_servers=1
		total_time_min = 1440
		now = time.localtime(time.time())
		out_str = str(now.tm_year) + str(now.tm_mon) +  str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
		for i in range(0,num_nodes):
			node_root = os.path.join(curpath,root_name,str(i))
			output_name = str(i) + '_output_' + out_str
			bashCommand = os.path.join(self.homepath,'t_slurm_md ') + str(num_servers) + ' ' + str(total_time_min) + ' ' + output_name + ' ' + node_root
			print time.strftime(self.time_format,time.gmtime()) + ' ' + hostname + ' job Creation: ' + bashCommand
			os.system(bashCommand)

	def create_bashCommand_train(self,type_node, node, input_path, str_ps, str_workers, index):
		root = self.homepath
		if type_node == 'ps':
             	     bashCommand = 'ssh-keygen -R '+node+';ssh -f -n -E log' +node+ ' ' + node + ' "cd ' + root + '/;unset http_proxy;unset https_proxy;stdbuf -i0 -o0 -e0 nohup ' + self.python_dir + ' ./main_md.py --phase=train --input_path=' + self.dirtrain + ' --beam_size=3 --node_root=' + root + ' --ps_hosts=' + str_ps + ' --worker_hosts=' + str_workers + ' --task_index=' + str(index) + ' --job_name=ps --distributed=True > nohup_' + node + ' "'
		else:
			bashCommand = 'ssh-keygen -R '+node+';ssh -f -n -E log' +node+ ' ' + node + ' "cd ' + root + '/;unset http_proxy;unset https_proxy;stdbuf -i0 -o0 -e0 nohup ' + self.python_dir + ' ./main_md.py --phase=train --input_path=' + input_path + ' --beam_size=3 --node_root=' + root + ' --ps_hosts=' + str_ps + ' --worker_hosts=' + str_workers + ' --task_index=' + str(index) + ' --job_name=worker --distributed=True > nohup_' + node + ' "'
		return bashCommand

	def create_jobs_train(self,num_nodes):
                
		print 'Jobs Train: ' + str(self.list_dirs)	
		curpath = self.curpath
		cluster, index_task_ps, index_task_workers = self.cluster, self.index_task_ps,self.index_task_workers
	
		num_ps = len(cluster['ps'])
		print str(cluster['ps'])
		str_ps = ','.join([x for x in cluster['ps']])
		input_path_ps = self.list_dirs[0]
	
		print str(cluster['worker'])
		num_worker = len(cluster['worker'])
		str_workers = ','.join([x for x in cluster['worker']])
		input_path_worker = self.list_dirs[0:num_worker]

        	now = time.localtime(time.time())
        	out_str = str(now.tm_year) + str(now.tm_mon) +  str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
	
		print 'Parameter Servers Preparation'
		print 'Number of parameter servers:' + str(num_ps)
       	 	for i in range(0,num_ps):
			server = self.cluster['ps'][i][0:6]
			print 'Parameter Server name: ' + str(server)
                	bashCommand = self.create_bashCommand_train('ps',server,input_path_ps,str_ps,str_workers,index_task_ps[i])
			status = os.system(bashCommand)
			print time.strftime(self.time_format,time.gmtime()) + ' ' + hostname + ' status: ' + str(status) + ' job Creation: ' + bashCommand

		print 'Worker Servers Preparation'
		print self.cluster
		print 'Number fo workers:' + str(num_worker)
		for i in range(0,num_worker):
                	server = self.cluster['worker'][i][0:6]
			print 'Worker Server name: ' + str(server)
			bashCommand = self.create_bashCommand_train('worker',server,input_path_worker[i],str_ps,str_workers,index_task_workers[i])
			os.system(bashCommand)
			print time.strftime(self.time_format,time.gmtime()) + ' ' + hostname + ' status: ' + str(status) + ' job Creation: ' + bashCommand

if __name__ == '__main__':
	#To train the data add the 3 parameters # of workers nodes, train and # of Parameter Servers
	config = Config()
	curpath = config.dest_path
	testimagespath = config.test_image_dir
	trainimagespath= config.train_image_dir
	python_dir = config.python_dir
	params = sys.argv
	hostname = '[' + os.environ['HOSTNAME'] +']'
	num_params = len(params)
	if num_params != 4:
		print 'input correct number of parameters'
		print 'Number of nodes, type of mode (train or infer) and number of parameter servers'
		sys.exit(2)
	print ' Node numbers: ' + str(params[1])
	num_nodes = int(params[1]) if int(params[1]) != 0 else 5
	type_mode = params[2]
	ps_nodes = int(params[3]) if int(params[3]) != 0 else 2
	worker_nodes = num_nodes - ps_nodes 
	dp = DataPreparation(ps_nodes,curpath,testimagespath,trainimagespath,type_mode,python_dir)
	dp.homepath = os.getcwd()	

	if type_mode == 'infer':
	        root_name = 'nodes' + str(num_nodes)
		root = os.path.join(curpath,root_name)
		if not os.path.exists(root):
			os.mkdir(root) 
			dp.create_dir(num_nodes,type_mode,root_name)
		dp.create_jobs_inference(num_nodes,root_name)

	if type_mode == 'train':
		root_name = 'train_nodes' + str(num_nodes)
		root = os.path.join(curpath,root_name)
                if not os.path.exists(root):
                	os.mkdir(root)
			dp.create_dir(worker_nodes,type_mode,root_name)
		else:
			dp.data_prepared = True
                lst = next(os.walk(root))[1]
                lst.sort()
                dp.list_dirs= [os.path.join(root,x,'images') for x in lst]			
		dp.create_jobs_train(num_nodes)
