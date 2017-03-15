% classify MNIST by linear SVM
% 2017-3-15 09:36:34

clear,clc;

% Prior steps:
% MNIST data: http://yann.lecun.com/exdb/mnist/
% mnistHelper: http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
% LIBSVM: https://github.com/cjlin1/libsvm
% C++ compiler: type "mex -setup c++" to see if it exists

% unzip data and toolbox
gunzip('*.gz');
unzip('mnistHelper.zip');

% LIBSVM
unzip('libsvm-master.zip');
cd libsvm-master/matlab/;
fprintf('MEX... \n');
make;
fprintf('\n');
cd ../../;
addpath(genpath('libsvm-master/'));

% load data
x_train=loadMNISTImages('train-images-idx3-ubyte');
x_test=loadMNISTImages('t10k-images-idx3-ubyte');
y_train=loadMNISTLabels('train-labels-idx1-ubyte');
y_test=loadMNISTLabels('t10k-labels-idx1-ubyte');

% transpose
x_train=x_train';
x_test=x_test';

% linear SVM
tic;
options='-t 0 -q';
model=svmtrain(y_train, x_train, options);
[y_predict,accuracy,prob_estimates]=svmpredict(y_test, x_test, model);
fprintf('Time: %0.2f minutes. \n', toc/60);

% % Output:
% Accuracy = 93.98% (9398/10000) (classification)
% Time: 13.37 minutes. 