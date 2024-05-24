# HW_QCNN
Simulation for Hamming Weight Preserving QCNN

## Guide - How to test QCNN models

### Test in the LIP6 Cluster interactively

The cluster server we use is called **Convergence**, you can check the website https://front.convergence.lip6.fr/convergence_en.html or my guide.

you can open a terminal and connect the server by:
```bash
(base) letao@ppqi29-wifi ~ % ssh letao@front.convergence.lip6.fr
Last login: Wed Apr 24 16:31:05 2024 from 132.227.92.49
(base) letao@front:~$
```
Here "letao" is my account name, you should replace it by yours.

After connected, please install the packages we need, at least pytorch, scipy, numpy, etc (e.g. you can use pip install),
and clone this git

```bash
git clone git@github.com:ptitbroussou/HW_QCNN.git
```
Then alloc the server resource by:
```bash
(base) letao@front:~$ salloc --nodes=1 --gpus-per-node=a100_3g.40gb:1 --time=60
salloc: Granted job allocation 11162
salloc: Nodes node04 are ready for job
```
Here "time=60" means we use this computational node for 60 minutes, you can change it.

Now you can open a new terminal and execute:
```bash
(base) letao@ppqi29-wifi ~ % ssh -J letao@front.convergence.lip6.fr letao@node04.convergence.lip6.fr
Last login: Wed Apr 24 13:02:04 2024 from 2001:660:3302:28c7:6efe:54ff:fe4e:8a18
(base) letao@node04:~$
```



and execute the file you want 

```bash
(base) letao@node04~$ cd HW_QCNN/Cluster_files
(base) letao@node04:~/HW_QCNN/Cluster_files$ python QCNN_3D.py 
Epoch 0: Loss = 2.227023, accuracy = 10.0000 %
Epoch 1: Loss = 2.185051, accuracy = 10.0000 %
Epoch 2: Loss = 2.135189, accuracy = 30.0000 %
Epoch 3: Loss = 2.104200, accuracy = 30.0000 %
Epoch 4: Loss = 2.095546, accuracy = 30.0000 %
Epoch 5: Loss = 2.079498, accuracy = 30.0000 %
Epoch 6: Loss = 2.086181, accuracy = 30.0000 %
Epoch 7: Loss = 2.062258, accuracy = 30.0000 %
Epoch 8: Loss = 1.997192, accuracy = 30.0000 %
Epoch 9: Loss = 2.031348, accuracy = 30.0000 %
Evaluation on test set: Loss = 2.278623, accuracy = 14.0000 %
```

### Test in the LIP6 Cluster non-interactively
You can create a "batch.sh" file in the fold "Cluster_files" with content
```
#!/bin/bash

#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --constraint=amd
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=80G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=1500
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

python QCNN_3D.py
```
Here we can change:
* job-name=run: name shown in the squeue
* mem=80G: depends on GPU we use
* gpus=a100_7g.80gb:1: you can also use a100_3g.40gb, less waiting time
* time=1500: 1500min, means this program will stop in 25h

Then execute this file
```bash
(base) letao@front:~$ cd new/HW_QCNN/Cluster_files
(base) letao@front:~/new/HW_QCNN/Cluster_files$ sbatch batch.sh
Submitted batch job 13588
(base) letao@front:~/new/HW_QCNN/Cluster_files$ 
```
You can use the command "squeue" to check if your task is running (sometimes you need to wait). After the task finished (or time out), you can check a output file (or error file) in the current fold.
If you use "a100_7g.80gb", maybe it takes a lot of time to wait, you can change it to "a100_3g.40gb" or use the interactive way.


### Others


There are two folders here, the files in the **Jupyter_test** you can use for your own testing in the IDE.
And the files in the **Cluster_files** you can run directly in (Cluster) terminal, these files in two folds has the same content.


Little tips: you can use VS Code to ssh connect the server, it's a easy way to edit server files.

## Detailed guide in the LIP6 website
https://front.convergence.lip6.fr/convergence_en.html
