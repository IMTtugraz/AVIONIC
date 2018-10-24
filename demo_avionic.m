%% demo script for matlab based avionic reconstruction
clear all; close all hidden; clc
addpath(genpath('./matlab/'))
unix(['export PATH=',pwd,'/CUDA/bin/:${PATH}']);

%% 1) dynamic Cartesian MRI reconstruction 2d-time with ICTGV, spatio-temp. TGV or spatio-temp. TV

% simulation parameter
acc     = 8;
pattern = 'VISTA';   % 'VRS', 'UIS'
method  = 'ICTGV2'; % 'TGV2', 'TV'
use_gpu = 1;

% get rawdata
unix(['wget https://zenodo.org/record/815385/files/testdata_cart_cine_avionic.mat --no-check-certificate']);
load testdata_cart_cine_avionic


[nslices,m,ncoils,nframes]  = size(data);

% simulate undersampling
% (requires VISTA: https://github.com/osu-cmr/vista) in matlab path
mri_obj.mask = simulate_pattern(nslices,m,nframes,acc,pattern);
mri_obj.data = data.*permute(repmat(mri_obj.mask,[1 1 1 ncoils]),[1 2 4 3]);

% reconstruction
if use_gpu
    [ recon_avionic, comp1, comp2, b1, u0, pdgap, datanorm, ictgvnorm, datafid] = ...
        avionic_matlab_gpu( mri_obj, {'method',method;'scale',1});
    
else
    
    switch method
        case 'ICTGV2'
            [recon_avionic,comp1,comp2,par,b1,tvt,gap,g2_out,sig_out,tau_out] = ...
                ictgv2_dmri(data,{});

        case 'TGV2'
            [recon_avionic,par,b1,tvt,gap,g2_out,sig_out,tau_out] = ...
                tgv2_dmri(data, {});
            
        case 'TV'
            [recon_avionic,par,b1,tvt,gap] = ...
                tv_dmri(data, {});
            
    end
end

% display results
implay(vidnorm(abs(recon_avionic)));


%% 2)   static Cartesian MRI reconstruction 3D volume with TGV 
%       Dataset: Cartesian VIBE (partial Fourier), simulate additional CAIPIRINHA undersampling

unix(['wget https://zenodo.org/record/815385/files/testdata_cart_avionic_tgv3d.mat --no-check-certificate' ]);
load testdata_cart_avionic_tgv3d.mat

acc          = 6;       % acceleration factor
shift        = 3;       % caipi-shift
aclines      = [24,24]; % auto-calibration lines in center
mri_obj.mask = caipirinha([size(mri_obj.data)],aclines,acc,shift);

recon_tgv3d_cart ...
             = avionic_matlab_gpu( mri_obj,...
             {'method','TGV2_3D';'stop_par',500;'isnoncart',0;'lambda',2});


% display results
implay(vidnorm(abs(recon_tgv3d_cart)));

%% 3)   static Non-Cartesian MRI reconstruction 3D volume with TGV 
%       Dataset: radial VIBE

unix(['wget https://zenodo.org/record/815385/files/testdata_noncart_avionic_tgv3d.mat --no-check-certificate']);
load testdata_noncart_avionic_tgv3d.mat

% dataset acquired with 550 spokes (fully sampled equals 400 spokes)
nspokes_acc = 21;

[nRO,nspokes,nslices,ncoils]= size(rawdata(:,1:nspokes_acc,:,:));

col                 = @(x) x(:);

osf = 2; % oversampling
wg  = 3; % kernel width
sw  = 16; % parallel sectors
FT2 = gpuNUFFT([col(traj(:,1:nspokes_acc,:,1)),col(traj(:,1:nspokes_acc,:,2)),...
    col(traj(:,1:nspokes_acc,:,3))]',ones(size((dcf(:)))),...
    osf,wg,sw,[nRO/2,nRO/2,nslices],[],true);

for coil=1:ncoils    
   img(:,:,:,coil )= FT2'*col(rawdata(:,1:nspokes_acc,:,coil).*(dcf(:,1:nspokes_acc,:))); 
end


triv_recon = sum(conj(b1).*img,4);

mri_obj.data        = reshape((rawdata(:,1:nspokes_acc,:,:))...
                      ,[nRO nspokes_acc*nslices 1 ncoils]);
mri_obj.traj        = [ col(traj(:,1:nspokes_acc,:,1)),...
                        col(traj(:,1:nspokes_acc,:,2)),...
                        col(traj(:,1:nspokes_acc,:,3))];
mri_obj.dcf         = reshape(dcf(:,1:nspokes_acc,:),[nRO,nspokes_acc*nslices]);
mri_obj.b1          = b1;
[n,m,l,ncoils]      = size(mri_obj.b1);
mri_obj.imgdims     = [n,m,l];
mri_obj.u0          = zeros(n,m,l)+1i.*zeros(n,m,l);


recon_tgv3d_noncart = ...
                 avionic_matlab_gpu( mri_obj,...
                {'method','TGV2_3D';'stop_par',500;...
                'isnoncart',1;'dx',1;'dy',1;'dz',3;'scale',1});


% display results
implay(vidnorm(abs(recon_tgv3d_noncart)));



