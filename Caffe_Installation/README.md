# Caffe Installation on Ubuntu
<p align="left"> <img src="https://raw.githubusercontent.com/GKalliatakis/Adventures-in-deep-learning/master/logo.png?raw=true" /> </p>

### Overview
Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (BVLC) and by community contributors.

---


### Installation:
First, you need to install the general dependencies using the following command:
```sh
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
```

After installing these, you need to install three more dependencies:
- CUDA is required for GPU mode. Installing CUDA is optional but it is recommended for better performance. To install CUDA, you can visit the link https://developer.nvidia.com/cuda-downloads  and then you can download CUDA for Ubuntu 14.04. Start the download and go get some coffee since it is quite large and will take time to download. After successful download, install it. 
- BLAS : Install ATLAS by `sh sudo apt-get install libatlas-base-dev` or install OpenBLAS or MKL for better CPU performance.
- BOOST : BOOST C++ library can be downloaded and installed through [Sourceforge](https://sourceforge.net/projects/boost/files/boost/1.58.0/).
- OpenCV 2.4 or above: For installing OpenCV, follow this [link](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html).
- Python: If you use the default python, then you will need to `sudo apt-get install python-dev` package.
### Compilation:
Now you need to edit the config file (Makefile.config). Create the config file by copying the contents of Makefile.config.example file to Makefile.config using the following command:
```sh cp Makefile.config.example  Makefile.config ```

After this, you need to change the configurations in Makefile.config file. Change the config according to the following conditions:
- For cuDNN acceleration, you should uncomment the USE_CUDNN := 1 switch in Makefile.config.
- For CPU-only Caffe, uncomment CPU_ONLY := 1 in Makefile.config.

The complete configuration of my Makefile can also be found on the current repo.

After making successful changes in the configuration file, you need to run the following commands:

```sh 
make all
make test
make runtest
```

(Before running next two compilation commands, you need to make sure that you have set PYTHON and MATLAB path in Makefile.config)
To compile the PYTHON wrappers, you need to run the command,
```sh 
make pycaffe
```

To compile the MATLAB wrappers, you need to run the command ,
```sh 
make metacaffe
```

Finally, if you have reached here, then you have installed Caffe on your System successfully.  

License
----

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [@thomasfuchs]: <http://twitter.com/thomasfuchs>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [keymaster.js]: <https://github.com/madrobby/keymaster>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]:  <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
