clear all
close all
clc

%% addpath
addpath(genpath('./matlab'))  %%set path to AVIONIC/matlab directory

%% Load 
% US data
data = importdata('ExampleDataASLTGV.mat');
[rNy,rNx,rNz,ncoils,nscan] = size(data);

% coil sensitivities
cmap = importdata('CoilSens.mat');

%% Prepare for TGV time recon
C_y2z2 = data(:,:,:,:,2:2:end);
L_y2z2 = data(:,:,:,:,1:2:end);

% Define US mask
mask_y2z2 = ones(size(C_y2z2));
mask_y2z2(C_y2z2 == 0) = 0;

% Define Parameters
method = 'ASLTGV2RECON4D';
alpha1 = 1.0;
w = 1.73;
N = 12;
lambda_c = 7;
tsw = 7;
alpha = 0.9;
lambda_l = 1.06*lambda_c;

par_in = {'maxIt','1000';...
          'lambda_c',num2str(lambda_c);...
          'lambda_l',num2str(lambda_l);...
          'alpha',num2str(alpha);...
          'alpha1','1.0';...
		  'alpha0',num2str(w);...
	      'dx','1.0';...
	      'dy','1.0';...
	      'dz','1.0';...
          'dt','1.0';...
          'timeSpaceWeight',num2str(tsw);...
          'gpu_device','3'};

% Run ASL-TGV reconstruction
[C_y2z2_tgvtime,L_y2z2_tgvtime] = asl_matlab_gpu(squeeze(C_y2z2(:,:,:,:,1:N)).*mask_y2z2(:,:,:,:,1:N), squeeze(L_y2z2(:,:,:,:,1:N)).*mask_y2z2(:,:,:,:,1:N), method, par_in, mask_y2z2(:,:,:,:,1:N), cmap);  

% Calculate PWI
PWI_tgv = abs(C_y2z2_tgvtime)-abs(L_y2z2_tgvtime);
PWI_mean_tgv = mean(PWI_tgv,4);

% Plot PWI
figure;
montage(permute(flip(PWI_mean_tgv,3),[2 1 4 3]),'DisplayRange',[0 max(PWI_mean_tgv(:))/4])
