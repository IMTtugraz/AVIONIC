function [ g2, comp1, comp2, b1, u0, pdgap, datanorm, ictgvnorm, datafid ] = avionic_matlab_gpu( mri_obj, par_in)
% simple export and import to gpu from matlab

%Regularization parameters for coil construction
method = 'ICTGV2';
stop_par = 500;
parfile = './CUDA/config/default.cfg';

lambda = 5.804591;
alpha0 = 1.41421356;
alpha1 = 1;
alpha = 0.5;
timeSpaceWeight2 = 0.5;
timeSpaceWeight = 4;
dx = 1;
dy = 1;
dz = 1;
dt = 1;

scale = 0;
isnoncart = 0;

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

% set zero output
comp1 = 0; comp2 = 0; pdgap = 0; u0 = 0;
prescale = 0; ictgvnorm = 0; datafid = 0; b1=0;

eval(['mkdir ',id]);


if scale
    scalestr = ' -o ';
else
    scalestr = ' ';
end

%--------------------------------------------------------------------
% define method
%---------------------------------------------------------------------
switch method
    case 'ICTGV2'
        voxelscalestr = [...
            ' --ictgv2.dx=',num2str(dx),' --ictgv2.dy=',num2str(dy),...
            ' --ictgv2.dt=',num2str(dt)];
        methodstr = [...
            ' --ictgv2.lambda=',num2str(lambda),...
            ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
            ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
            ' --ictgv2.alpha=',num2str(alpha),' '];
    case 'TGV2'
        voxelscalestr = ...
            [' --tgv2.dx=',num2str(dx),' --tgv2.dy=',num2str(dy),...
            ' --tgv2.dt=',num2str(dt)];
        methodstr = [...
            ' --tgv2.lambda=',num2str(lambda),...
            ' --tgv2.timeSpaceWeight=',num2str(timeSpaceWeight),' '];
    case 'TV'
        voxelscalestr = ...
            [' --tv.dx=',num2str(dx),' --tv.dy=',num2str(dy),...
            ' --tv.dt=',num2str(dt)];
        methodstr = [...
            ' --tv.lambda=',num2str(lambda),...
            ' --tv.timeSpaceWeight=',num2str(timeSpaceWeight),' '];
    case 'TGV2_3D'
        voxelscalestr = ...
            [' --tgv2_3D.dx=',num2str(dx),' --tgv2_3D.dy=',num2str(dy),...
            ' --tgv2_3D.dz=',num2str(dz)];
        methodstr = [...
            ' --tgv2_3D.lambda=',num2str(lambda),' '];
end
%------------------------------------------------------------------

%------------------------------------------------------------------
% CARTESIAN OR NON-CARTESIAN
%------------------------------------------------------------------
if isnoncart
    
    if strcmp(method,'TGV2_3D') % 3d
        [nreadouts, nencodings1, nencodings2, ncoils] = size(mri_obj.data);
        n = mri_obj.imgdims(1);
        m = mri_obj.imgdims(2);
        l = mri_obj.imgdims(3);
        nframes = l;
        % data
        writebin_vector(mri_obj.data,...
            ['./',id,'/data.bin']);
        
        % trajectory
        % split traj in x1 x2 x3 .... y1 y2 y3 .... z1 z2 z3 .... order (needed by gpuNUFFT!)
        kk = [(mri_obj.traj(:,1)), (mri_obj.traj(:,2)), mri_obj.traj(:,3)];
        writebin_vector(kk,...
            ['./',id,'/mask.bin']);
        
        % density compensation
        writebin_vector((mri_obj.dcf(:)),...
            ['./',id,'/dcf.bin']);
        
        
        b1u0str = exportb1u0(mri_obj,id,[1 2 3 4],[1 2 3]);
        
        
        dimstr = [  num2str(m),':',num2str(n),':',num2str(l),':',...
            num2str(nreadouts),':',num2str(nencodings1),':',num2str(nencodings2),':',...
            num2str(ncoils),':1 '];
        
    else %2d-time
        [nsamplesonspoke, nspokes, ncoils, nframes] = size(mri_obj.data);
        n = mri_obj.imgdims(1);
        m = mri_obj.imgdims(2);
        
        wscale=1;
        
        % data
        writebin_vector(reshape(mri_obj.data.*wscale,[nspokes*nsamplesonspoke ncoils nframes]),...
            ['./',id,'/data.bin']);
        
        % trajectory
        % split traj in x1 x2 x3 .... y1 y2 y3 order (needed by gpuNUFFT!)
        kk = reshape([real((mri_obj.traj(:))), imag((mri_obj.traj(:)))],[nspokes*nsamplesonspoke*nframes 2]);
        kk = reshape(kk,[nspokes*nsamplesonspoke nframes 2]);
        
        writebin_vector(permute(kk,[1 3 2]),...
            ['./',id,'/mask.bin']);
        
        % density compensation
        writebin_vector((mri_obj.dcf(:)),...
            ['./',id,'/dcf.bin']);
        
        
        b1u0str = exportb1u0(mri_obj,id,[2 1 3],[2 1]);
        
        dimstr = [  num2str(m),':',num2str(n),':0:',...
            num2str(nsamplesonspoke),':',num2str(nspokes),':0:',...
            num2str(ncoils),':',num2str(nframes),' '];
    end
    trajstr= [ '-n --dens ./',id,'/dcf.bin '];
    
    %=====================
else % cartesian data
    %=====================
    if strcmp(method,'TGV2_3D') % 3d
        
        [n,m,l,ncoils] = size(mri_obj.data);
        nframes = l;
        
           % setup data        
        mri_obj = setup_data3d(mri_obj);
      
        writebin_vector(permute(mri_obj.data,[2 1 3 4]),...
            ['./',id,'/data.bin']);
        writebin_vector(permute(mri_obj.mask,[2 1 3]),...
            ['./',id,'/mask.bin']);
          
        b1u0str = exportb1u0(mri_obj,id,[2 1 3 4],[2 1 3]);
        
        dimstr=[' ',num2str(m),':',num2str(n),':',num2str(l),':',...
            num2str(m),':',num2str(n),':',num2str(l),':',...
            num2str(ncoils),':0 '];
    else % 2d-time
        
        [n,m,ncoils,nframes] = size(mri_obj.data);
        
        % setup data
        mri_obj = setup_data(mri_obj);
        
        writebin_vector(permute(mri_obj.data,[2 1 3 4]),...
            ['./',id,'/data.bin']);
        writebin_vector(permute(mri_obj.mask,[2 1 3]),...
            ['./',id,'/mask.bin']);
        
        
        b1u0str = exportb1u0(mri_obj,id,[2 1 3],[2 1]);
        
        dimstr=[' ',num2str(m),':',num2str(n),':0:',...
            num2str(m),':',num2str(n),':0:',...
            num2str(ncoils),':',num2str(nframes),' '];
    end
    trajstr = '';
end
%------------------------------------------------------------------

if strcmp(method,'TGV2_3D')
    reshapeg2 = @(x) permute(reshape(x,[m,n,nframes]),[1 2 3]);
    reshapeb1 = @(x) permute(reshape(x,[m,n,nframes,ncoils]),[1 2 3 4]);
    reshapeu0 = @(x) permute(reshape(x,[m,n,nframes]),[1 2 3]);
    
else
    
    reshapeg2 = @(x) permute(reshape(x,[m,n,nframes]),[2 1 3]);
    reshapeb1 = @(x) permute(reshape(x,[m,n,ncoils]),[2 1 3]);
    reshapeu0 = @(x) permute(reshape(x,[m,n]),[2 1]);
end


%------------------------------------------------------------------
% Define Recon Command
%------------------------------------------------------------------
recon_cmd=['avionic ',scalestr, ' -i ',num2str(stop_par),...
    methodstr,...
    voxelscalestr,...
    ' -m ',method,' -e -p ',parfile,' -d ', dimstr,...
    b1u0str,trajstr,' ./',id,'/data.bin ./',id,'/mask.bin ./',id,'/result.bin'];


display(recon_cmd);

% run reconstruction
unix(recon_cmd);


% read results
g2 = readbin_vector(['./',id,'/result.bin']);
g2 = reshapeg2(g2);

if exist(['./',id,'/x3_component'])==2
    comp2 = readbin_vector(['./',id,'/x3_component']);
    comp2 = reshapeg2(comp2);
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
    b1 = reshapeb1(b1);
end

if exist(['./',id,'/u0_reconstructed.bin'])==2
    u0 = readbin_vector(['./',id,'/u0_reconstructed.bin']);
    u0 = reshapeu0(u0);
end

if exist(['./',id,'/datanorm_factor.bin'])==2
    datanorm = readbin_vector(['./',id,'/datanorm_factor.bin']);
end



% clean up
unix(['rm -rf ./',id])

end


function b1u0str = exportb1u0(mri_obj,id,orderb1,orderu0)


%------------------------------------------------------------------
% B1 Given or not
%------------------------------------------------------------------
if isfield(mri_obj,'b1');
    %writebin_vector((mri_obj.b1),...
    %  ['./',id,'/b1.bin']);
    writebin_vector(permute(mri_obj.b1,orderb1),...
        ['./',id,'/b1.bin']);
    %writebin_vector(permute(mri_obj.b1,[1 2 3 4]),...
    %     ['./',id,'/b1.bin']);
    
    if isfield(mri_obj,'u0');
        %writebin_vector((mri_obj.u0),...
        %    ['./',id,'/u0.bin']);
        writebin_vector(permute(mri_obj.u0,orderu0),...
            ['./',id,'/u0.bin']);
        %writebin_vector(permute(mri_obj.u0,[2 1 3]),...
        %    ['./',id,'/u0.bin']);
    end
    b1u0str = [' -u ./',id,'/u0.bin -s ./',id,'/b1.bin '];
else
    b1u0str = '';
end

end