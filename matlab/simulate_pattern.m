function [ spattern ] = simulate_pattern(n,m,nframes,acc,type)
% adapted from: https://github.com/osu-cmr/vista
% 
%
% Generates VISTA sampling pattern
% Version 1.0
% Last updated 08/01/2014
% Author: Rizwan Ahmad (ahmad.46@osu.edu)
% Reference: R. Ahmad, H. Xue, S. Giri, Y. Ding, J. Craft, O.P. Simonetti, 
% Variable Density Incoherent Spatiotemporal Acquisition (VISTA) 
% for Highly Accelerated Cardiac Magnetic Resonance Imaging, Magnetic 
% Resonance in Medicine, in revision
% 
% 2014, Patent Pending
%
% =========================================================================
% Copyright (c) 2014 - The Ohio State University.
% All rights reserved.
% 
% Permission to use, copy, modify, and distribute this software and its
% documentation for educational, research, and not-for-profit purposes,
% without fee and without written agreement, is hereby granted, provided 
% that the above copyright notice, the following two paragraphs, and the
% author attribution appear in all copies of this software. For commercial 
% licensing possibilities, contact The Office of Technology 
% Commercialization Office (http://tco.osu.edu/) at The Ohio State 
% University.
% 
% IN NO EVENT SHALL THE OHIO STATE UNIVERSITY BE LIABLE TO ANY PARTY 
% FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES 
% ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF 
% THE OHIO STATE UNIVERSITY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH 
% DAMAGE.
% 
% THE OHIO STATE UNIVERSITY SPECIFICALLY DISCLAIMS ANY WARRANTIES 
% INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
% AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS 
% ON AN "AS IS" BASIS, AND THE OHIO STATE UNIVERSITY HAS NO OBLIGATION TO 
% PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
% =========================================================================
%% Select a sampling type =================================================
param.typ = type;

% check frequency and phase encoding direction; usaually nFE > nPH with
% frequency oversampling and less phase-encoding steps

if n>m % first dimension is freqency endcoding dir
    nFE = n;
    nPH = m;
else
    nFE = m;
    nPH = n;
end

%% Select appropriate parameter values ====================================
param.p     = nPH; % Number of phase encoding steps
param.t     = nframes; % Number of frames
param.R     = acc;  % Net acceleration rate
param.alph  = 0.28;      % 0<alph<1 controls sampling density; 0: uniform density, 1: maximally non-uniform density
param.sig   = param.p/5; % Std of the Gaussian envelope for sampling density
param.sd    = 'shuffle';%10; % Seed to generate random numbers; a fixed seed should reproduce VISTA


%% Probably you don't need to mess with these paramters ===================
% If unsure, leave them empty and the default value will be employed.
param.nIter= []; % Number of iterations for VISTA (defualt: 120)
param.ss   = []; % Step-size for gradient descent. Default value: 0.25; 
param.tf   = []; % Step-size in time direction wrt to phase-encoding direction; use zero for constant temporal resolution. Default value: 0.0
param.s    = []; % Exponent of the potenital energy term. Default value 1.4
param.g    = []; % Every gth iteration is relocated on a Cartesian grid. Default value: floor(param.nIter/6)
param.uni  = []; % At param.uni iteration, reset to equivalent uniform sampling. Default value: floor(param.nIter/2)
param.W    = []; % Scaling of time dimension; frames are "W" units apart. Default value: max(param.R/6,1)
param.sz   = []; % Display size of samples. Default value: 3.5
param.dsp  = 5; % Display frequency (verbosity), every dsp-th iteration will be displayed. Default value: 1
param.fs   = 1; % Does time average has to be fully sampled, 0 for no, 1 for yes. Only works with VISTA. Default value: 1
param.fl   = []; % Start checking fully sampledness at fl^th iteration. Default value: floor(param.nIter*5/6)

                 
%% Check parameters values
param = checkParam(param);

%% Call VISTA to compute the 2D sampling pattern
samp = VISTA(param);
spattern = repmat(samp,[1 1 nFE]);

if n>m % first dimension is freqency endcoding dir
    spattern = permute(spattern,[3 1 2]);    
else
    spattern = permute(spattern,[1 3 2]);
end

end