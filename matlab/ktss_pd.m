function [x,fval,datanorm] = ktss_pd(mri_obj, ld,maxiter,ref,gpu)
%LS_PD:
% Primal-Dual implementation of the k-t-SPARSE-SENSE reconstruction for
% non-cartesian data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fval = 0;

n = mri_obj.imgdims(1);
m = mri_obj.imgdims(2);
nframes = size(mri_obj.data,4);
ncoils = size(mri_obj.data,3);
[nRO,nspf,~,~] = size(mri_obj.data);

Nd = mri_obj.imgdims;
Kd = floor(Nd*1.5);
n_shift = Nd/2;

if gpu
    Nd      = [n,m];
    osf     = 1.5; % oversampling: 1.5 1.25
    wg      = 6; % kernel width: 5 7
    sw      = 8; % parallel sectors' width: 12 16
    
    for frame=1:nframes
        FT{frame} = gpuNUFFT([  real(col(mri_obj.traj(:,:,frame))), ...
            imag(col(mri_obj.traj(:,:,frame)))]',...
            ones(nRO*nspf,1),osf,wg,sw,[n,m],[]);
    end
    
    K = @(x) backward_opt_noncart_gpu(x, mri_obj.b1, mri_obj.dcf, FT);
    Kh = @(x) forward_opt_noncart_gpu(x, mri_obj.b1, mri_obj.dcf, FT);
else
    
    for frame=1:nframes
        om = [ real(col(mri_obj.traj(:,:,frame))), ...
            imag(col(mri_obj.traj(:,:,frame)))]*2*pi;
        nufft_st{frame} = nufft_init(om,Nd,[6,6],Kd,n_shift,'kaiser');
    end
    mri_obj.nufft_st = nufft_st;
    
    K = @(x) backward_opt_noncart(x, mri_obj.nufft_st, mri_obj.b1, mri_obj.dcf);
    Kh = @(x) forward_opt_noncart(x, mri_obj.nufft_st, mri_obj.b1, mri_obj.dcf);
end

    
mri_obj = prepare_data_noncart(mri_obj, {'imgdims',mri_obj.imgdims(1:2);'J',[6,6]},1);

datanorm = mri_obj.datanorm;


% estimate operator norm using power iteration
x1  = rand(n,m,nframes);
y1  = Kh(K(x1));
for i=1:10
    if norm(y1(:))~=0
        x1 = y1./norm(y1(:));
    else
        x1 = y1;
    end
    [y1] = Kh(K(x1));
    l1 = y1(:)'*x1(:);
    fprintf('.');
end

opnorm = max(abs(l1)); 


L2      = 8;
sig     = 1/sqrt(L2+opnorm);
tau     = sig;

% primal variables
x       = zeros(n,m,nframes); 

% dual variables
y       = zeros(n,m,nframes);
z       = zeros(size(mri_obj.data));

ext     = x;

for k = 1:maxiter

    % dual update
    z = z + sig*(K(ext));
    y = y + sig*fgrad_t(ext);
  
        % proximal maps
        y = y./max(1,sqrt( y.^2 ));
        z = (z-sig*mri_obj.data ) / (1+sig/ld);

    % primal update
    ext = x - tau*(Kh(z) - bdiv_t(y));
    
    % extragradient update
    x = 2*ext - x;

    %Swap extragradient and primal variable
    [x,ext] = deal(ext,x);
    
     %Adapt stepsize
  %   if (k<10) || (rem(k,50) == 0) 
  %       [sig,tau] = steps_ls_pd(ext-x,sig,tau,K);
  %       display(['sig=',num2str(sig),' | tau=',num2str(tau)]);
  %   end
     
     if ~isempty(ref) && (rem(k,10)==0)
        fval(k) = norm(col(ref-x));
        display(['iter: ',num2str(k),' | fval=',num2str(fval(k))]);
     end
    
end


end

