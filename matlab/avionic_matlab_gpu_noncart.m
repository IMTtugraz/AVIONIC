function [ g2, comp1, comp2, b1, u0, pdgap, datanorm, ictgvnorm, datafid ] = avionic_matlab_gpu_noncart( mri_obj, par_in )
% simple export and import to gpu from matlab


%Regularization parameters for coil construction
method = 'ICTGV2';
stop_par = 500;
parfile = './stand_functions/default.cfg';

adapt_lambda = 0;
lambda = 5.804591;
alpha0 = 1.41421356;
alpha1 = 1;
alpha = 0.5;
timeSpaceWeight2 = 0.5;
timeSpaceWeight = 4;


%Read parameter-------------------------------------------------------------------------
%Input: par_in--------------------------------------------------------------------------
%Generate list of parameters
vars = whos;
for l=1:size(vars,1)
    par_list{l,1} = vars(l).name;
end
%Set parameters according to list
for l=1:size(par_in,1);
    valid = false;
    for j=1:size(par_list,1); if strcmp(par_in{l,1},par_list{j,1})
            valid = true;
            eval([par_in{l,1},'=','par_in{l,2}',';']);
        end; end
    if valid == false; warning(['Unexpected parameter at ',num2str(l)]); end
end
%---------------------------------------------------------------------------------------
%---------------------------------------------------------------------------------------

id = num2str(round(now*1e5));
unix(['mkdir ./',id]);

% set zero output
comp1 = 0; comp2 = 0; pdgap = 0; b1 = 0; u0 = 0;
ictgvnorm = 0; datafid = 0;


[nsamplesonspoke, nspokes, ncoils, nframes] = size(mri_obj.data);
n = mri_obj.imgdims(1);
m = mri_obj.imgdims(2);

%wscale = sqrt(permute(repmat(mri_obj.dcf,[1 1 1 ncoils]),[1 2 4 3]));
wscale = 1;

% data
writebin_vector(reshape(mri_obj.data.*wscale,[nspokes*nsamplesonspoke ncoils nframes]),...
    ['./',id,'/data.bin']);

% trajectory
% split traj in x1 x2 x3 .... y1 y2 y3 order (needed by gpuNUFFT!)
kk = reshape([real((mri_obj.traj(:))), imag((mri_obj.traj(:)))],[nspokes*nsamplesonspoke*nframes 2]);
kk = reshape(kk,[nspokes*nsamplesonspoke nframes 2]);

writebin_vector(permute(kk,[1 3 2]),...
    ['./',id,'/k.bin']);

% density compensation
writebin_vector((mri_obj.dcf(:)),...
    ['./',id,'/dcf.bin']);

if isfield(mri_obj,'b1');
    writebin_vector((mri_obj.b1),...
        ['./',id,'/b1.bin']);
    
    if isfield(mri_obj,'u0');
        writebin_vector((mri_obj.u0),...
            ['./',id,'/u0.bin']);
    end
    
    recon_cmd=['avionic -n -o -i ',num2str(stop_par),...
        ' --ictgv2.lambda=',num2str(lambda),...
        ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
        ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
        ' --ictgv2.alpha=',num2str(alpha),...
        ' -m ',method,' -e -p ',parfile, ...
        ' -s ./',id,'/b1.bin -u ./',id,'/u0.bin ', ...
        ' -d ',num2str(m),':',num2str(n),':0:',...
        num2str(nsamplesonspoke),':',num2str(nspokes),':0:',...
        num2str(ncoils),':',num2str(nframes),...
        ' --dens ./',id,'/dcf.bin ./',id,'/data.bin ./',id,'/k.bin ./',id,'/result.bin'];
    
else
    
    recon_cmd=['avionic -v -n -o  -i ',num2str(stop_par),...
        ' --ictgv2.lambda=',num2str(lambda),...
        ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
        ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
        ' --ictgv2.alpha=',num2str(alpha),...
        ' -m ',method,' -e -p ',parfile,' -d ', ...
        num2str(m),':',num2str(n),':0:',...
        num2str(nsamplesonspoke),':',num2str(nspokes),':0:',...
        num2str(ncoils),':',num2str(nframes),...
        ' --dens ./',id,'/dcf.bin ./',id,'/data.bin ./',id,'/k.bin ./',id,'/result.bin'];
end




display(recon_cmd);

% run reconstruction
unix(recon_cmd);

% read results
g2 = readbin_vector(['./',id,'/result.bin']);
g2 = permute(reshape(g2,[m,n,nframes]),[1 2 3]);

if exist(['./',id,'/x3_component'])==2
    comp2 = readbin_vector(['./',id,'/x3_component']);
    comp2 = permute(reshape(comp2,[m,n,nframes]),[1 2 3]);
    comp1 = g2-comp2;
end

if exist(['./',id,'/PDGap'])==2
    pdgap = readbin_vector(['./',id,'/PDGap']);
    pdgap = abs(pdgap);
end

if exist(['./',id,'/DATAfidelity'])==2
    datafid = readbin_vector(['./',id,'/DATAfidelity']);
    datafid = abs(datafid);
end

if exist(['./',id,'/ICTGVnorm'])==2
    ictgvnorm = readbin_vector(['./',id,'/ICTGVnorm']);
    ictgvnorm = abs(ictgvnorm);
end

if exist(['./',id,'/b1_reconstructed.bin'])==2
    b1 = readbin_vector(['./',id,'/b1_reconstructed.bin']);
    b1 = permute(reshape(b1,[m,n,ncoils]),[1 2 3]);
end

if exist(['./',id,'/u0_reconstructed.bin'])==2
    u0 = readbin_vector(['./',id,'/u0_reconstructed.bin']);
    u0 = permute(reshape(u0,[m,n]),[1 2]);
end

if exist(['./',id,'/datanorm_factor.bin'])==2
    datanorm = readbin_vector(['./',id,'/datanorm_factor.bin']);
end



% clean up
unix(['rm -rf ./',id])

