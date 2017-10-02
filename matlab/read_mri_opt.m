function [mri_obj,ref,sos,b1_full,datanorm] = read_mri_opt(filename,par_in)

%Set parameter#############################################################
addpath(genpath('/home2/ictgv/MRI_motion/workspace/MRI_motion/data/matlab/'));

% Standard parameters for artifial subsampling on self-generated datasets
crop = 0;                   % crop data to accelerate calculations
crop_factor = 31;           % crop factor

acc = 0;                          % acceleration factor
refL = 0;                         % reference lines or block in k-space center
sampling_pattern = 'randomcart';  % sampling scheme {'cart','randomcart','randomcart_xy','vista','vrs','uis'}

coil_cmp = 0;               % apply coil-compression (not tested)
coil_threshold = 0.05;      % defines threshold for coil reduction
coil_cmp_method = 'dacl';   % defines coil compression method


b1_code_refsos = 'walsh';          % estimate coil-sensitivities from full data for reference
phase_sensitive_refsos = 0;

%Regularization parameters for coil construction
u_reg = 0.2;
b1_reg = 2;
b1_final_reg = 0.1;
b1_final_nr_it = 1000;

b1_source=0;
normalize_data = 1;

is_perfusion = 0;

synthetic_coils = 0;

removeOS=0;

datasize = [128,128,10,15];

%Read parameter------------------------------------------------------------
%Input: par_in-------------------------------------------------------------
%Generate list of parameters
vars = whos;
for i=1:size(vars,1)
    par_list{i,1} = vars(i).name;
end
%Set parameters according to list
for i=1:size(par_in,1);
    valid = false;
    for j=1:size(par_list,1); if strcmp(par_in{i,1},par_list{j,1})
            valid = true;
            eval([par_in{i,1},'=','par_in{i,2}',';']);
        end; end
    if valid == false; warning(['Unexpected parameter at ',num2str(i)]); end
end


% Read dataset#############################################################
if ischar(filename)
    switch filename(end-2:end)
        
        % (1) ISMRM Challenge h5 datafile--------------------------------------
        case '.h5'
            display(['reading ' filename]);
            [kdata,idx,header]=read_ismrm_challenge_data(...
                ['/home/dieheart/workspace/MRI_motion/data/matlab/' filename]);
            
            %Save data in object
            mri_obj.data = double(kdata);
            mri_obj.mask = idx;
            mri_obj.b1 = 0;
            mri_obj.header = header;
            
            clear kdata idx header
            
            % for further consitent save porpuses
            filename = filename(1:end-3);
            display('data already subsampled - no additional subsampling performed!')
            
            acc = 0; %make sure that no additional subsampling is performed
            
            % (2) Self-generated test-cases----------------------------------------
            % subsampling parameters ARE considered, data must always be saved as
            % mri_obj.data
        case 'mat'
            display(['reading ' filename]);
            load( ['/home2/ictgv/MRI_motion/data/matlab/' filename]);
            
            % for further consitent save porpuses
            filename = filename(1:end-4);
            
            
            if exist('kdata','var')
                mri_obj.data = permute(kdata,[1 2 4 3]);
                mri_obj.b1 = b1;
                clear kdata b1
            end
            
        case 'bin'
            display(['reading ' filename]);
            data = readbin_vector(filename,'double');
            mri_obj.data = permute(reshape(data,[datasize]),[2 1 3 4]);
            ref=0;sos=0;b1_full=0;datanorm=1;
            return;
    end
    
    % (3) Manipulate already constructed mri_obj---------------------------
elseif isstruct(filename)
    
    mri_obj = filename;
    
end

[n,m,ncoils,nframes] = size(mri_obj.data);
if removeOS
    cutOS = n/4+1:n*3/4; 
    fprintf('remove readout oversampling ...');
    %mri_obj.data = reshape(mri_obj.data,[n,m*ncoils*nframes]);
    databuf = zeros(n/2,m*ncoils*nframes);
    for i=1:size(databuf,2)
       
        rosh = ifft(mri_obj.data(:,i));
        rosh(cutOS) = [];%rosh(end/4:(3*end/4)+1,:);
        databuf(:,i) = fft(rosh);
        
    end
    mri_obj.data = reshape(databuf,[n/2,m,ncoils,nframes]);
    
    fprintf('done');
end
mri_obj.data(isnan(mri_obj.data)) = 0;
[n,m,ncoils,nframes] = size(mri_obj.data);
mri_obj.data = double(mri_obj.data);
mri_obj.mask = (squeeze(abs(mri_obj.data(:,:,1,:)))~=0);



% Estimate sensitivities###################################################
%   for reference reconstruction from full data
display('estimate coil-sensitivites from full data for reference reconstruction')

if phase_sensitive_refsos
    if strcmp(b1_code_refsos,'const');
        display('Using coil_construction code for b1 field...');
        mri_obj_coilest = setup_data(mri_obj);
        [b1_full,~] = coil_construction_opt(mri_obj_coilest.data,mri_obj_coilest.mask,...
            {'u_reg',u_reg;'b1_reg',b1_reg;'b1_final_reg',b1_final_reg;'b1_final_nr_it',b1_final_nr_it});
        clear mri_obj_coilest;
        
    elseif strcmp(b1_code_refsos,'given');
        display('Using given b1 field...');
        b1_full = b1_source;
        
    elseif strcmp(b1_code_refsos,'walsh')
        display('Using walsh code...');
        % estimate b1_walsh for comparison from time-averaged data as
        % in " Otazo et al, Low-rank plus sparse matrix decomposition for accelerated dynamic MRI
        % with separation of background and dynamic components, 2013"
        mri_obj_est = setup_data(mri_obj);
        [~,crec] = get_sos_crec_opt(mri_obj_est.data,mri_obj_est.mask);
        [~,b1_full] = walsh_sens_2d(crec);
        clear coil_est_data crec
        
    else
        
        error('Wrong b1 code');
        
    end
else
    b1_full = 0;
end
% Modulate synthetic sensitivities#############################################################
% use above b1 maps for generation of phase-consistent reference
% reconstruction and modulate syntetic sensitivities
if synthetic_coils
    %%
    display('modulate syntetic coil sensitivities...');
    if phase_sensitive_refsos
        error('must set phase_sensitive_refsos = 1');
    end
    FOV_x = n/1000;
    FOV_y = m/1000;
    
    % for future use these parameters (better independancy of sens.)
    if m>n
        coil_radius = .06*FOV_y/0.28; %.07*FOV_y/0.28
        coil_distfromcenter = .17*FOV_y/0.28; %.17*FOV_y/0.28;
    elseif m<= n
        coil_radius = .06*FOV_x/0.28; %.07*FOV_y/0.28
        coil_distfromcenter = .17*FOV_x/0.28; %.17*FOV_y/0.28;
    end
    
    % generate synthetic sensitivity maps using biot-savarts law
    b1_synt = GenerateSensitivityMap( [FOV_x,FOV_y],[FOV_x/n, FOV_y/m], ncoils, coil_radius, coil_distfromcenter);
    b1_synt_scale = sqrt(sum(abs(b1_synt).^2,3));
    b1_synt = b1_synt./repmat(b1_synt_scale,[1 1 size(b1_synt,3)]);
    
    % make frame-wise reference reconstruction using
    % sensitivites estimated as above
    for frame = 1:nframes
        for coil=1:ncoils
            refhelp(:,:,coil) = fft2c(mri_obj.data(:,:,coil,frame)).*sqrt(n*m);
        end
        ref(:,:,frame) = sum(conj(b1_full).*refhelp,3);
        % ref2(:,:,frame) = sqrt(sum(abs(refhelp).^2,3));
    end
    
    % normalize phase to "brightest" coil (as for walsh and opt)
    [mx,pos] = max( squeeze(sum(sum(abs(repmat(sum(ref(:,:,1),3)/nframes,[1 1 ncoils]).*b1_synt),1),2)));
    angcor = angle(b1_synt(:,:,pos));
    b1_synt = b1_synt.*repmat(exp(-1i.*angcor),[1 1 ncoils]);
    
    % generate new synthetic data with synthetic coils
    for frame = 1:nframes
        for coil = 1:ncoils
            data(:,:,coil,frame) = ifft2c(ref(:,:,frame).*b1_synt(:,:,coil));
        end
    end
    
    % scale equal to original data
    data = data.*(max(abs(mri_obj.data(:)))/max(abs(data(:))));
    
    
    %                     % check  new reference
    %                      for frame = 1:nframes
    %                           for coil=1:ncoils
    %                              refhelp3(:,:,coil) = fft2c(data(:,:,coil,frame)).*sqrt(n*m);
    %                           end
    %                         refafter(:,:,frame) = sqrt(sum(abs(refhelp3).^2,3));
    %                      end
    %                      % refafter = refafter.*sqrt(n*m);
    %
    %                     % check coil dependancy
    %                      datacheck1 = reshape(squeeze(data(:,:,:,1)),[n*m,ncoils]);
    %                      datacheck2 = reshape(squeeze(mri_obj.data(:,:,:,1)),[n*m,ncoils]);
    %                      [~,S1,~] = svd(datacheck1,'econ');
    %                      [~,S2,~] = svd(datacheck2,'econ');
    %                      figure; hold on
    %                      plot(diag(S1),'-or');plot(diag(S2),'-ob');
    %                      legend('synthetic','true')
    %                   imagine(b1_full,ref,refafter);
    
    mri_obj.data = data;
    b1_full = b1_synt;
    clear ref refhelp data b1_synt
    
end

% make reference reconstruction#############################################################
fprintf('make reference reconstruction:  ')
if phase_sensitive_refsos
    fprintf(['b1 weighted combination - ',b1_code_refsos]);
else
    fprintf(['sum-of-squares combination ']);
end

ref = zeros(n,m,nframes);
for frame=1:nframes
    refhelp = zeros(n,m,ncoils);
    for coil=1:ncoils
        % use fft2c --> same result as fft2(chop ... ) and faster
        refhelp(:,:,coil) = fft2c(mri_obj.data(:,:,coil,frame)).*sqrt(n*m);
        %refhelp(:,:,coil) = fft2(ifftshift(chop.*mri_obj.data(:,:,coil,frame)));%./sqrt(n*m);
    end
    
    if phase_sensitive_refsos
        % b1 weighted combination (Roemer PB et al, The NMR phased array. Magn Reson Med 1990;16:192–225.)
        % and rescale
        ref(:,:,frame) = sum(conj(b1_full).*refhelp,3);
    else
        
        % sum-of-squares combination
        ref(:,:,frame) = sqrt(sum(abs(refhelp).^2,3));
    end
    
end


% Preprocess dataset#######################################################

% coil compression-------------------------------------------------
if coil_cmp
    
    %Get coil-wise reconstruction
    dcrec = zeros(n,m,ncoils);
    for j = 1:ncoils
        data = sum( squeeze(mri_obj.data(:,:,j,:)), 3);
        msum = sum(mri_obj.mask,3);
        data(msum>0) = data(msum>0)./msum(msum>0);
        dcrec(:,:,j) = data;
    end
    
    dcrec = reshape(dcrec,[n*m,ncoils]);
    [U,S,V] = svd(dcrec,'econ');
    
    ncoils_svd = find(diag(S)/S(1)>coil_threshold, 1, 'last' ); % 10
    
    figure; hold on; plot(diag(S));stem(ncoils_svd,max(diag(S)),'r');
    
    display(['perform coil compression - ',num2str(ncoils_svd),'/',num2str(ncoils),' virtual coils'])
    
    
    for frame = 1:nframes
        kdata_ = reshape(mri_obj.data(:,:,:,frame),[n*m,ncoils]);
        kdata_ = reshape(kdata_*V(:,1:ncoils_svd),n*m,ncoils_svd);
        kdata_svd(:,:,:,frame) = reshape(kdata_,[n,m,ncoils_svd]);
    end
    mri_obj.data = kdata_svd;
    mri_obj.mask = (squeeze(abs(mri_obj.data(:,:,1,:)))~=0);
    ncoils = ncoils_svd;
    clear kdata_svd U S V
    
    % estimate new sensitivites from compressed data
    if strcmp(b1_code_refsos,'const');
        display('Using coil_construction code for b1 field...');
        mri_obj_coilest = setup_data(mri_obj);
        [b1_full,~] = coil_construction_opt(mri_obj.data,mri_obj.mask,...
            {'u_reg',u_reg;'b1_reg',b1_reg;'b1_final_reg',b1_final_reg;'b1_final_nr_it',b1_final_nr_it});
        clear mri_obj_coilest;
        
    elseif strcmp(b1_code_refsos,'given');
        display('Using given b1 field...');
        b1_full = b1_source;
        
    elseif strcmp(b1_code_refsos,'walsh')
        display('Using walsh code...');
        % estimate b1_walsh for comparison from time-averaged data as
        % in " Otazo et al, Low-rank plus sparse matrix decomposition for accelerated dynamic MRI
        % with separation of background and dynamic components, 2013"
        mri_obj_est = setup_data(mri_obj);
        [~,crec] = get_sos_crec_opt(mri_obj_est.data,mri_obj_est.mask);
        [~,b1_full] = walsh_sens_2d(crec);
        clear coil_est_data crec
    else
        
        error('Wrong b1 code');
        
    end
    
end %end coil_cmp


% define accelerated sampling pattern------------------------------
if acc ~= 0
    
    % if sampling pattern already exists just load it
    if exist(['/home2/ictgv/MRI_motion/stand_functions/sampling_pattern/',sampling_pattern,num2str(n),num2str(m),num2str(nframes),num2str(acc),'.mat']) == 2;
        
        display(['loading sampling pattern: ',sampling_pattern,': acc = ',num2str(acc),' / refL = ' num2str(refL)])
        eval(['load /home2/ictgv/MRI_motion/stand_functions/sampling_pattern/',sampling_pattern,num2str(n),num2str(m),num2str(nframes),num2str(acc),'.mat']);
    else
        
        display(['generate new sampling pattern: ',sampling_pattern,': acc = ',num2str(acc),' / refL = ' num2str(refL)])
        
        switch sampling_pattern
            
            case 'cart'%...............................................
                % home made
                spattern = zeros(n,m,nframes);
                if n>m % first dimension is freqency endcoding dir
                    for t=1:nframes
                        spattern(:,mod(t,acc)+1:acc:end,t) = 1;
                    end
                    % add reference lines
                    if refL ~=0
                        spattern(:,floor((end-refL)/2):floor((end+refL)/2),:) = 1;
                    end
                else % second dimension is frequency endcoding dir
                    for t=1:nframes
                        spattern(mod(t,acc)+1:acc:end,:,t) = 1;
                    end
                    % add reference lines
                    if refL ~=0
                        spattern(floor((end-refL)/2):floor((end+refL)/2),:,:) = 1;
                    end
                end
                
                
            case 'randomcart'%.........................................
                % from ESMRMB Workshop on Parallel Imaging, wuerzburg 2012, sebastian kozerke, eth zuerich
                if n>m % first dimension is freqency endcoding dir
                    spattern = GenerateSamplingPattern(acc,[n,m,nframes]);
                    spattern = repmat(spattern,[n 1 1]);
                    % maybe use
                    %[M3 M2] = Hybrid_DownsamplingMASK(nY, nX, nframe, down)
                    
                else
                    spattern = GenerateSamplingPattern(acc,[m,n,nframes]);
                    spattern = permute(repmat(spattern,[m 1 1]),[2 1 3]);
                    
                end
                
            case 'randomcart_xy'%......................................
                %MAGMA. Feb 2011; 24(1): 43–50.
                %doi:  10.1007/s10334-010-0234-7
                %Adapted Random Sampling Patterns for Accelerated MRI
                %Florian Knoll, Christian Clason, Clemens Diwoky, and Rudolf Stollberger
                spdf = genPDF([n,m],10,1/acc,2,0,0);
                for frame=1:nframes
                    [spattern(:,:,frame),R_true] = gen_random_pattern(spdf,acc,6);
                end
                
                % add reference lines
                if refL ~=0
                    spattern(floor((end-refL)/2):floor((end+refL)/2),floor((end-refL)/2):floor((end+refL)/2),:) = 1;
                end
                
            case 'vista'
                % R. Ahmad, H. Xue, S. Giri, Y. Ding, J. Craft, O.P. Simonetti,
                % Variable Density Incoherent Spatiotemporal Acquisition (VISTA)
                % for Highly Accelerated Cardiac Magnetic Resonance Imaging, Magnetic
                % Resonance in Medicine, in revision
                spattern = vista_pattern(n,m,nframes,acc,'VISTA');
            case 'vrs'
                spattern = vista_pattern(n,m,nframes,acc,'VRS');
            case 'uis'
                spattern = vista_pattern(n,m,nframes,acc,'UIS');
            otherwise
                error('no valid sampling pattern')
        end
        save(['/home2/ictgv/MRI_motion/stand_functions/sampling_pattern/',sampling_pattern,num2str(n),num2str(m),num2str(nframes),num2str(acc),'.mat'],'spattern')
        
    end
    
    
else % no acceleration or already subsampled data
    
    mri_obj.mask = (squeeze(abs(mri_obj.data(:,:,1,:)))~=0);
    spattern = mri_obj.mask;
    
end % end acc
  
% mask data ----------------------------------------------------------
mri_obj.data = mri_obj.data.*permute(repmat(spattern,[1 1 1 ncoils]),[1 2 4 3]);
mri_obj.mask = spattern;


% crop for accelerated calculation---------------------------------
if crop
    display('Cropping...')
    mri_obj.data = mri_obj.data( floor(n/2-crop_factor):floor(n/2+crop_factor),...
        floor(m/2-crop_factor):floor(m/2+crop_factor),1:floor(ncoils/2),1:floor(nframes/2) );
    mri_obj.mask = mri_obj.mask( floor(n/2-crop_factor):floor(n/2+crop_factor),...
        floor(m/2-crop_factor):floor(m/2+crop_factor),1:floor(nframes/2) );
    [n,m,ncoils,nframes] = size(mri_obj.data);
    
end % end crop


% normalize and mask data------------------------------------------
% ATTENTION: for future scale data to 1 but for ISMRM datasets we need to consider old scaling from *.h5 datafiles
if normalize_data
    
    if is_perfusion
        % normalize to time-averaged data
        fmask = zeros(n,m);
        hamsize = 40;
        f = hamming(hamsize) * hamming(hamsize)';
        fmask((n/2-hamsize/2)+1:(n/2+hamsize/2),(m/2-hamsize/2)+1:(m/2+hamsize/2)) = f;
        crec = zeros(n,m,ncoils);
        for j = 1:ncoils
            data = sum( squeeze(mri_obj.data(:,:,j,:)), 3);
            msum = sum(mri_obj.mask,3);
            data(msum>0) = data(msum>0)./msum(msum>0);
            crec(:,:,j) = fft2c( data.*fmask ).*sqrt(n*m);
        end
        u = sqrt( sum( abs(crec).^2 , 3) );
        % normalize to sum-of-square dynamic data
        sos = zeros(n,m,nframes);
        for j = 1:nframes
            data = ( squeeze(mri_obj.data(:,:,:,j)));
            sos(:,:,j) = sqrt(sum(abs(fft2c( data ).*sqrt(n*m)).^2,3));
        end
        [a,b] = hist(sos(:),100);
        [a1,b1] = hist(u(:),100);
        
       % figure; imshow(u,[]); colorbar(); title(num2str(max(abs(u(:)))));
        
        %datanorm = 255./median(sos(sos>=0.9.*max(sos(:))));
        datanorm =  255/1.3725./median(u(u>=0.9.*max(u(:))));
        %datanorm = 255./median(u(u>=0.95.*max(u(:))));
        %datanorm = 255./(max(u(:)));
        %figure; hold on;
        %subplot(1,2,1);
        %bar(b1,a1);
        %stem(median(u(u>=0.9.*max(u(:)))),max(a1));
        %stem(median(u(u>=0.95.*max(u(:)))),max(a1),'r');
        %subplot(1,2,2);bar(b1,a1);
        
    else
        crec = zeros(n,m,ncoils);
        for j = 1:ncoils
            data = sum( squeeze(mri_obj.data(:,:,j,:)), 3);
            msum = sum(mri_obj.mask,3);
            data(msum>0) = data(msum>0)./msum(msum>0);
            crec(:,:,j) = fft2c( data ).*sqrt(n*m);
        end
        u = sqrt( sum( abs(crec).^2 , 3) );
        datanorm = 255./median(u(u>=0.9.*max(u(:))));
    end
    
    clear u crec
    mri_obj.data = mri_obj.data.*datanorm;
    display(['scale data to [0 255] - datanorm = ',num2str(datanorm)]);
    
    % rescale reference with datanorm
    ref = ref.*datanorm;
else
    datanorm = 1;
    warning(['no scaling!']);
end

% casting to double because of
% Error using  *  MTIMES is not supported for one sparse input and one single input.
% Error in sp_grad_3_1 (line 7)
mri_obj.mask = double(mri_obj.mask);
mri_obj.data = double(mri_obj.data);

% check acceleration
display(['check acc: ', num2str(numel(mri_obj.mask)./sum(mri_obj.mask(:)))]);


% make non phase-sensitive sos reconstruction###############################
sos = zeros(n,m,nframes);
for frame=1:nframes
    soshelp = zeros(n,m,ncoils);
    for coil=1:ncoils
        % use fft2c --> same result as fft2(chop ... ) and faster
        soshelp(:,:,coil) = fft2c(mri_obj.data(:,:,coil,frame)).*sqrt(n*m);
    end
    
    % sum-of-squares combination
    sos(:,:,frame) = sqrt(sum(abs(soshelp).^2,3));
end


end % end read_mri_opt




%Helper Functions----------------------------------------------------------
%--------------------------------------------------------------------------
function [smp] = GenerateSamplingPattern(R,dims)
% from esmrmb workshop on parallel imaging, sebastian kozerke, eth zuerich

if R>1
    rad = abs(linspace(-1,1,dims(2)));
    
    minval = 0;
    maxval = 1;
    frac   = floor(1/R*dims(2));
    
    while (1)
        val = minval/2+maxval/2;
        pdf = (1-rad).^R+val; pdf(find(pdf>1)) = 1;
        tot = floor(sum(pdf(:)));
        if tot > frac
            maxval = val;
        end
        if tot < frac
            minval = val;
        end
        if tot == frac
            break;
        end
    end
    
    pdf(find(pdf>1)) = 1;
    pdfsum = sum(pdf(:));
    
    smp = single(zeros(1,dims(2),dims(3)));
    
    for t=1:dims(3)
        tmp = zeros(size(pdf));
        while abs(sum(tmp(:))-pdfsum) > 1
            tmp = rand(size(pdf))<pdf;
        end
        smp(1,:,t) = tmp;
    end
else
    smp = ones(1,dims(2),dims(3));
end
end


%--------------------------------------------------------------------------
function [pattern,R_True] = gen_random_pattern(spdf,R,wind)
% INPUT:
% pdf:      sampling pdf
% R:        desired reduction factor
% wind:     window size for filling of holes
%
% OUTPUT
% pattern:   sampling pattern
% R_True:    true reduction factor

%% generate optimized random subsampling pattern
[N,M] = size(spdf);

% normalize pdf (int(pdf)=1)
spdf = spdf/sum(spdf(:));
P   = spdf(:);

% cumulative distribution
P   = cumsum(P) ./ sum(P);
S   = floor(length(P)/R); % safety margin since indices can be redrawn

% try several random patterns, take most incoherent one
pattern = zeros(size(spdf));
minval  = 1e99;
maxit   = 3;

for n=1:maxit
    mask  = [];
    draws = S;
    while draws>0
        d_mask = zeros(draws,1);
        for i = 1:draws
            X = rand < P;
            d_mask(i) = sum(X);
        end
        d_mask = (length(P) + 1) - d_mask;
        mask   = unique([mask;d_mask]);
        draws  = S-length(mask);
    end
    
    try_pattern = zeros(size(spdf));
    try_pattern(mask) = 1;
    
    coherence = ifft2(try_pattern./spdf);
    max_coher = max(abs(coherence(2:end)));
    if  max_coher < minval
        minval  = max_coher;
        pattern = try_pattern;
    end
end

% fill holes
for i=1:N
    for j = 1:M
        imin = max(1,i-wind); imax = min(N,i+wind);
        jmin = max(1,j-wind); jmax = min(M,j+wind);
        if max(pattern(imin:imax,jmin:jmax)) < 1 % window empty
            pattern(i,j) = 1;
        end
    end
end

% effective acceleration factor
R_True = length(P)/nnz(pattern);
end


%--------------------------------------------------------------------------
function [pdf,val] = genPDF(imSize,p,pctg,distType,radius,disp)
%[pdf,val] = genPDF(imSize,p,pctg,distType,radius,disp])
%
%	generates a pdf for a 1d or 2d random sampling pattern
%	with polynomial variable density sampling
%
%	Input:
%		imSize - size of matrix or vector
%		p - power of polynomial
%		pctg - partial sampling factor e.g. 0.5 for half
%		distType - 1 or 2 for L1 or L2 distance measure
%		radius - radius of fully sampled center
%		disp - display output
%
%	Output:
%		pdf - the pdf
%		val - min sampling density
%
%
%	Example:
%	[pdf,val] = genPDF([128,128],2,0.5,2,0,1);
%
%	(c) Michael Lustig 2007

if nargin < 4
    distType = 2;
end

if nargin < 5
    radius = 0;
end

if nargin < 6
    disp = 0;
end


minval=0;
maxval=1;
val = 0.5;

if length(imSize)==1
    imSize = [imSize,1];
end

sx = imSize(1);
sy = imSize(2);
PCTG = floor(pctg*sx*sy);


if sum(imSize==1)==0  % 2D
    [x,y] = meshgrid(linspace(-1,1,sy),linspace(-1,1,sx));
    switch distType
        case 1
            r = max(abs(x),abs(y));
        otherwise
            r = sqrt(x.^2+y.^2);
            r = r/max(abs(r(:)));
    end
    
else %1d
    r = abs(linspace(-1,1,max(sx,sy)));
end


idx = find(r<radius);

pdf = (1-r).^p; pdf(idx) = 1;
if floor(sum(pdf(:))) > PCTG
    error('infeasible without undersampling dc, increase p');
end

% begin bisection
while(1)
    val = minval/2 + maxval/2;
    pdf = (1-r).^p + val; pdf(find(pdf>1)) = 1; pdf(idx)=1;
    N = floor(sum(pdf(:)));
    if N > PCTG % infeasible
        maxval=val;
    end
    if N < PCTG % feasible, but not optimal
        minval=val;
    end
    if N==PCTG % optimal
        break;
    end
end

if disp
    figure,
    subplot(211), imshow(pdf)
    if sum(imSize==1)==0
        subplot(212), plot(pdf(end/2+1,:));
    else
        subplot(212), plot(pdf);
    end
end

end


%--------------------------------------------------------------------------
function [kdata,idx,header] = read_ismrm_challenge_data(ismrmrdfile)

%%
%  This code reads the data for the ISMRM challenge into
%   matlab.
%
%   Input:
%       .h5 file name formatted by the ISMRM raw data format
%   Output:
%       kdata   raw complex k-space data
%       idx     locations of sample points
%       header  structure containing all entries from xml header

%%
%  Read the Header
%
header = ismrmrd_header2struct(h5read(ismrmrdfile,'/dataset/xml'));

% Pull Matrix size from header
Nx =  str2num( header.encoding.reconSpace.matrixSize.x.Text);
Ny =  str2num( header.encoding.reconSpace.matrixSize.y.Text);

%%
% Read the raw data
%
raw_data = h5read(ismrmrdfile,'/dataset/data');
number_acquisitions = numel(raw_data.traj);
disp(['number_acquisitions=',num2str(number_acquisitions)]);

%% These are data specific
Nframes = max(raw_data.head.idx.phase)+1;
Ncoils =  max(raw_data.head.active_channels);

disp(['Nx=',num2str(Nx),',Ny=',num2str(Ny),'Ncoils=',num2str(Ncoils),'Nframes=',num2str(Nframes)]);
%% Pre allocate data
kdata = zeros(Nx,Ny,Ncoils,Nframes);
idx = zeros(Nx,Ny,Nframes);


%% Now go through data
for acq = 1:number_acquisitions
    
    traj = raw_data.traj{acq};
    data= raw_data.data{acq};
    
    %Reshape Data to correct format
    data = reshape(data(1:2:end)+1i*data(2:2:end),[raw_data.head.active_channels(acq) raw_data.head.number_of_samples(acq)]);
    
    % Get Coordinates
    i = traj(1:raw_data.head.trajectory_dimensions(acq):end);
    j = traj(2:raw_data.head.trajectory_dimensions(acq):end);
    frame =  raw_data.head.idx.phase(acq);
    
    %Assign
    for pos=1:raw_data.head.number_of_samples(acq)
        idx(i(pos)+1,j(pos)+1,frame+1)=1;
        for coil =1:raw_data.head.active_channels(acq)
            kdata(i(pos)+1,j(pos)+1,coil,frame+1)=data(coil,pos);
        end
    end
    
end

return;

end


%--------------------------------------------------------------------------
function s = ismrmrd_header2struct(header)
%
%
%  s = ismrmrd_header2struct(header)
%
%  Converts and ISMRMRD xml header to a struct.
%

if (iscell(header)),
    header_string = header{1};
else
    header_string = header;
end

if (ischar(header_string) < 1),
    error('Malformed input header. Is not a character string.');
end

tmp_nam = tempname;
f=fopen(tmp_nam,'w');
fwrite(f,header_string);
fclose(f);

s = xml2structX(tmp_nam);
s  = s.ismrmrdHeader;
return

end


%--------------------------------------------------------------------------
function [ s ] = xml2structX( file )
%Convert xml file into a MATLAB structure
% [ s ] = xml2struct( file )
%
% A file containing:
% <XMLname attrib1="Some value">
%   <Element>Some text</Element>
%   <DifferentElement attrib2="2">Some more text</Element>
%   <DifferentElement attrib3="2" attrib4="1">Even more text</DifferentElement>
% </XMLname>
%
% Will produce:
% s.XMLname.Attributes.attrib1 = "Some value";
% s.XMLname.Element.Text = "Some text";
% s.XMLname.DifferentElement{1}.Attributes.attrib2 = "2";
% s.XMLname.DifferentElement{1}.Text = "Some more text";
% s.XMLname.DifferentElement{2}.Attributes.attrib3 = "2";
% s.XMLname.DifferentElement{2}.Attributes.attrib4 = "1";
% s.XMLname.DifferentElement{2}.Text = "Even more text";
%
% Please note that the following characters are substituted
% '-' by '_dash_', ':' by '_colon_' and '.' by '_dot_'
%
% Written by W. Falkena, ASTI, TUDelft, 21-08-2010
% Attribute parsing speed increased by 40% by A. Wanner, 14-6-2011
% Added CDATA support by I. Smirnov, 20-3-2012
%
% Modified by X. Mo, University of Wisconsin, 12-5-2012

if (nargin < 1)
    clc;
    help xml2struct
    return
end

if isa(file, 'org.apache.xerces.dom.DeferredDocumentImpl') || isa(file, 'org.apache.xerces.dom.DeferredElementImpl')
    % input is a java xml object
    xDoc = file;
else
    %check for existance
    if (exist(file,'file') == 0)
        %Perhaps the xml extension was omitted from the file name. Add the
        %extension and try again.
        if (isempty(strfind(file,'.xml')))
            file = [file '.xml'];
        end
        
        if (exist(file,'file') == 0)
            error(['The file ' file ' could not be found']);
        end
    end
    %read the xml file
    xDoc = xmlread(file);
end

%parse xDoc into a MATLAB structure
s = parseChildNodes(xDoc);

return

end


%--------------------------------------------------------------------------
% ----- Subfunction parseChildNodes -----
function [children,ptext,textflag] = parseChildNodes(theNode)
% Recurse over node children.
children = struct;
ptext = struct; textflag = 'Text';
if hasChildNodes(theNode)
    childNodes = getChildNodes(theNode);
    numChildNodes = getLength(childNodes);
    
    for count = 1:numChildNodes
        theChild = item(childNodes,count-1);
        [text,name,attr,childs,textflag] = getNodeData(theChild);
        
        if (~strcmp(name,'#text') && ~strcmp(name,'#comment') && ~strcmp(name,'#cdata_dash_section'))
            %XML allows the same elements to be defined multiple times,
            %put each in a different cell
            if (isfield(children,name))
                if (~iscell(children.(name)))
                    %put existsing element into cell format
                    children.(name) = {children.(name)};
                end
                index = length(children.(name))+1;
                %add new element
                children.(name){index} = childs;
                if(~isempty(fieldnames(text)))
                    children.(name){index} = text;
                end
                if(~isempty(attr))
                    children.(name){index}.('Attributes') = attr;
                end
            else
                %add previously unknown (new) element to the structure
                children.(name) = childs;
                if(~isempty(text) && ~isempty(fieldnames(text)))
                    children.(name) = text;
                end
                if(~isempty(attr))
                    children.(name).('Attributes') = attr;
                end
            end
        else
            ptextflag = 'Text';
            if (strcmp(name, '#cdata_dash_section'))
                ptextflag = 'CDATA';
            elseif (strcmp(name, '#comment'))
                ptextflag = 'Comment';
            end
            
            %this is the text in an element (i.e., the parentNode)
            if (~isempty(regexprep(text.(textflag),'[\s]*','')))
                if (~isfield(ptext,ptextflag) || isempty(ptext.(ptextflag)))
                    ptext.(ptextflag) = text.(textflag);
                else
                    %what to do when element data is as follows:
                    %<element>Text <!--Comment--> More text</element>
                    
                    %put the text in different cells:
                    % if (~iscell(ptext)) ptext = {ptext}; end
                    % ptext{length(ptext)+1} = text;
                    
                    %just append the text
                    ptext.(ptextflag) = [ptext.(ptextflag) text.(textflag)];
                end
            end
        end
        
    end
end
return


end


%--------------------------------------------------------------------------
% ----- Subfunction getNodeData -----
function [text,name,attr,childs,textflag] = getNodeData(theNode)
% Create structure of node info.

%make sure name is allowed as structure name
name = toCharArray(getNodeName(theNode))';
name = strrep(name, '-', '_dash_');
name = strrep(name, ':', '_colon_');
name = strrep(name, '.', '_dot_');

attr = parseAttributes(theNode);
if (isempty(fieldnames(attr)))
    attr = [];
end

%parse child nodes
[childs,text,textflag] = parseChildNodes(theNode);

if (isempty(fieldnames(childs)) && isempty(fieldnames(text)))
    %get the data of any childless nodes
    % faster than if any(strcmp(methods(theNode), 'getData'))
    % no need to try-catch (?)
    % faster than text = char(getData(theNode));
    text.(textflag) = toCharArray(getTextContent(theNode))';
end

return

end


%--------------------------------------------------------------------------
% ----- Subfunction parseAttributes -----
function attributes = parseAttributes(theNode)
% Create attributes structure.

attributes = struct;
if hasAttributes(theNode)
    theAttributes = getAttributes(theNode);
    numAttributes = getLength(theAttributes);
    
    for count = 1:numAttributes
        %attrib = item(theAttributes,count-1);
        %attr_name = regexprep(char(getName(attrib)),'[-:.]','_');
        %attributes.(attr_name) = char(getValue(attrib));
        
        %Suggestion of Adrian Wanner
        str = toCharArray(toString(item(theAttributes,count-1)))';
        k = strfind(str,'=');
        attr_name = str(1:(k(1)-1));
        attr_name = strrep(attr_name, '-', '_dash_');
        attr_name = strrep(attr_name, ':', '_colon_');
        attr_name = strrep(attr_name, '.', '_dot_');
        
        
        %KMJ Changed to make things values vs. text
        tmpSTR = str((k(1)+2):(end-1));
        if( sum(isletter(tmpSTR)==0) )
            attributes.(attr_name) = str2num(tmpSTR);
        else
            attributes.(attr_name) = tmpSTR;
        end
        %attributes.(attr_name) = str((k(1)+2):(end-1));
    end
end

return

end




