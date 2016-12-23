% demo script for matlab based avionic reconstruction
clear all; close all hidden; clc

% simulation parameter
acc = 12;
pattern = 'VRS';   % 'VRS', 'UIS'
method = 'ICTGV2'; % 'TGV2', 'TV'
use_gpu = 0;

% get rawdata
if ~exist(['./cardiac_cine_full.mat'])==2
    unix(['wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/cardiac_cine_full.mat']);
end
load cardiac_cine_full

[n,m,ncoils,nframes]  = size(data);

% simulate undersampling
% (requires VISTA: https://github.com/osu-cmr/vista) in matlab path
mask = simulate_pattern(n,m,nframes,acc,pattern);
data = data.*permute(repmat(mask,[1 1 1 ncoils]),[1 2 4 3]);

% reconstruction
if use_gpu
    [ g2, comp1, comp2, b1, u0, pdgap ] = ...
        avionic_matlab_gpu( data, {'method',method;});
else
    
    switch method
        case 'ICTGV2'
            [g2,comp1,comp2,par,b1,tvt,gap,g2_out,sig_out,tau_out] = ...
                ictgv2_dmri(data,{'stop_par',500;'debug',1});
            
            [g22,comp12,comp22,par2,b1,tvt2,gap2,g2_out2,sig_out2,tau_out2] = ...
                ictgv2_dmri_step(data, {'stop_par',500;'debug',1;...
                        'sig',1/3;...
                        'tau',1/3;...
                        'delta',1;...
                        'mu', 0.7;...
                        'beta', 1;...
                        'gamma',0.1;});
        case 'TGV2'
            [g2,par,b1,tvt,gap,g2_out,sig_out,tau_out] = ...
                tgv2_dmri(data, {});
            
        case 'TV'
            [g2,par,b1,tvt,gap] = ...
                tv_dmri(data, {});
            
    end
end

% display results
implay(vidnorm(abs(comp1)));





