function [C,L] = asl_matlab_gpu(data_c, data_l, method, par_in, mask, varargin)
% data_c            Control data size (x,y,z,coils,aver)
% data_l            Label data size(x,y,z,coils,aver)
% method            Select Method 'ASLTGV2RECON4D'
% par_in            struct containing the parameters
% par_in.lambda_c   regularization parameter control data
% par_in.lambda_l 	regularization parameter label data
% par_in.alpha0     weightening symetric divergence term (default = sqrt(3))
% par_in.alpha1     weightening gradient term (default = 1)
% par_in.alpha      weightening between the two TGV terms (default = 0.9)
% par_in.maxIt      number of iterations (default = 1000)
% par_in.dx         grid distances -isotropic resolution dA = [1 1 1]*1/dx)
%                                  -unisotropic resolution dA = [1/dx 1/dy 1/dz]
% par_in.dy
% par_in.dz
% par_ind.dt		grid distance in temporal direction (default = 1)
% mask				undersampling pattern if not defined mask is set to ones size (x,y,z,coils,aver)
% varargin{1} 		coil sensitivity maps

% Check input
if (nargin < 6 && strcmp(method,'ASLTGV2RECON'))
    error('Coil sensitivity maps are not defined')
elseif nargin < 5
    mask = ones(size(data_c));
elseif nargin < 4
    disp('Using default parameters')
elseif nargin < 2
    error('Not enough input data')
end

%Estimate memory
enough_memory = estimate_Memory(data_c);
disp(sprintf('Estimated GPU memory:%f MB',enough_memory))
if ~enough_memory
  error('Not enough GPU memory')
end

%Default parameters
    parfile = './CUDA/config/default.cfg';
    lambda_c = 10;
    lambda_l = 10;	
    alpha0 = sqrt(3);     
    alpha1 = 1;
    alpha = 0.9;  
    maxIt = 1000;
    dx = 1;
    dy = 1;
    dz = 1;
    dt = 1;
    timeSpaceWeight = 10;
    gpu_device = 0;

%% Read parameter
% Generate list of parameters
vars = whos;
for l = 1:size(vars,1)
    par_list{l,1} = vars(l).name;
end
% Set parameters according to list
for l = 1:size(par_in,1);
    valid = false;
    for j = 1:size(par_list,1);
        if strcmp(par_in{l,1},par_list{j,1})
            valid = true;
            eval([par_in{l,1},'=','par_in{l,2}',';']);
        end
    end
    if valid == false;
        warning(['Unexpected parameter at ', num2str(l)]);
    end
end

id = num2str(round(now*1e5+rand(1)*1E7));
eval(['mkdir ',strcat('ph',id)]);

%% Setup data
data_c = setup_data3dasl(data_c,mask);
[data_l,mask] = setup_data3dasl(data_l,mask);

%% Norm input data Control and Label image
norm_cl = getNorm(data_l,data_c,mask);
data_c = data_c/norm_cl;
data_l = data_l/norm_cl;

%% write data
writebin_vector(data_c,['./ph',id,'/data_c.bin']);
writebin_vector(data_l,['./ph',id,'/data_l.bin']);
if strcmp(method,'ASLTGV2RECON4D')
    mask = squeeze(mask(:,:,:,1,:));
end
writebin_vector(mask,['./ph',id,'/mask.bin']);

%% Setup command for ASL-TGV recon
if strcmp(method,'ASLTGV2RECON4D')
	    [x,y,z,coils,aver] = size(data_c);
    cmap = varargin{1};
    size(cmap)
    writebin_vector(cmap,['./ph',id,'/cmap.bin']);
	denoise_cmd = ['avionic ',...
        ' -i ',num2str(maxIt),...
        ' --asltgv2recon4D.lambda_l=', num2str(lambda_l),...
        ' --asltgv2recon4D.lambda_c=', num2str(lambda_c),...
		' --asltgv2recon4D.timeSpaceWeight=', num2str(timeSpaceWeight),...
        ' --asltgv2recon4D.alpha=', num2str(alpha),...
        ' --asltgv2recon4D.alpha1=', num2str(alpha1),...
        ' --asltgv2recon4D.alpha0=', num2str(alpha0),...
		' --asltgv2recon4D.dx=', num2str(dx),...
        ' --asltgv2recon4D.dy=', num2str(dy),...
        ' --asltgv2recon4D.dz=', num2str(dz),...
        ' --asltgv2recon4D.dt=', num2str(dt),...
        ' -b ', num2str(gpu_device),...
        ' -m ',method,' -p ',parfile,' -d ', num2str(x),...
        ':', num2str(y), ':', num2str(z), ':', num2str(x),...
        ':', num2str(y), ':', num2str(z), ':', num2str(coils), ':',...
         num2str(aver),...
        ' -s', './ph', id, '/cmap.bin', ' -l', ' ./ph', id,'/data_l.bin ./ph', id,...
        '/data_c.bin ./ph' ,id, '/mask.bin ./ph',id,'/result.bin'];	
else
    error('Undefined method')
end

disp(denoise_cmd)
    
%% Run recon
unix(denoise_cmd);

%% Read results
if(strcmp(method,'ASLTGV2RECON4D'))
   C_ = readbin_vector(['./ph',id,'/resultC.bin']);
   L_ = readbin_vector(['./ph',id,'/resultL.bin']);
   C = reshape(C_,[x y z aver])*norm_cl;
   L = reshape(L_,[x y z aver])*norm_cl;
else
   error('Undefined method')
end

%% Clean up
unix(['rm -rf ./ph',id])

end
