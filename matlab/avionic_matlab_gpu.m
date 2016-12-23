function [ g2, comp1, comp2, b1, u0, pdgap, datanorm ] = avionic_matlab_gpu( data, par_in, mask, b1)
% simple export and import to gpu from matlab

    if nargin < 4
        b1 = [];
    end

    if nargin < 3
        mask = squeeze(data(:,:,1,:))~=0;
    end

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
    comp1 = 0; comp2 = 0; pdgap = 0; 
    prescale = 0;

    [n,m,ncoils,nframes] = size(data);

    mri_obj.data = data;
    mri_obj.mask = mask;
    clear data mask

    % setup data
    mri_obj = setup_data(mri_obj);
    
    eval(['mkdir ',id]);
    % write data to binary file
  
    if ~isempty(b1);
        crec = zeros(n,m,ncoils);
        for j = 1:ncoils
            data_ = sum( squeeze(mri_obj.data(:,:,j,:)), 3);
            msum = sum(mri_obj.mask,3);
            data_(msum>0) = data_(msum>0)./msum(msum>0);
            crec(:,:,j) = fft2( data_ )./sqrt(n*m);
        end
        u0 = sum(crec.*conj(b1),3);
        %u0 = sqrt(sum(abs(crec).^2,3));
        datanorm_pre = 255./median(abs(u0(abs(u0)>=0.9.*max(abs(u0(:))))));
        mri_obj.data = mri_obj.data.*datanorm_pre;
        u0 = u0*datanorm_pre;
        prescale = 1;

        writebin_vector(permute(b1,[2 1 3 4]),...
            ['./',id,'/b1.bin']);
        writebin_vector(permute(u0,[2 1]),...
            ['./',id,'/u0.bin']);
        clear crec u

        recon_cmd=['avionic -v -o -i ',num2str(stop_par),...
            ' --ictgv2.lambda=',num2str(lambda),...
            ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
            ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
            ' --ictgv2.alpha=',num2str(alpha),...
            ' -m ',method,' -e -p ',parfile,' -d ', ...
            num2str(m),':',num2str(n),':0:',num2str(m),':',num2str(n),':0:',num2str(ncoils), ...
            ':',num2str(nframes),' -u ./',id,'/u0.bin -s ./',id,'/b1.bin ./',id,'/data.bin ./',id,'/mask.bin ./',id,'/result.bin'];
    else
        recon_cmd=['avionic -v -o -i ',num2str(stop_par),...
            ' --ictgv2.lambda=',num2str(lambda),...
            ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
            ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
            ' --ictgv2.alpha=',num2str(alpha),...
            ' -m ',method,' -e -p ',parfile,' -d ', ...
            num2str(m),':',num2str(n),':0:',num2str(m),':',num2str(n),':0:',num2str(ncoils), ...
            ':',num2str(nframes),' ./',id,'/data.bin ./',id,'/mask.bin ./',id,'/result.bin'];
    end

    display(recon_cmd);

    writebin_vector(permute(mri_obj.data,[2 1 3 4]),...
        ['./',id,'/data.bin']);
    writebin_vector(permute(mri_obj.mask,[2 1 3]),...
        ['./',id,'/mask.bin']);

    
    % run reconstruction
    unix(recon_cmd);

    % read results
    g2 = readbin_vector(['./',id,'/result.bin']);
    g2 = permute(reshape(g2,[m,n,nframes]),[2 1 3]);

    if exist(['./',id,'/x3_component'])==2
        comp2 = readbin_vector(['./',id,'/x3_component']);
        comp2 = permute(reshape(comp2,[m,n,nframes]),[2 1 3]);
        comp1 = g2-comp2;
    end

    if exist(['./',id,'/PDGap'])==2
        pdgap = readbin_vector(['./',id,'/PDGap']);
        pdgap = abs(pdgap);
    end

    if exist(['./',id,'/b1_reconstructed.bin'])==2
        b1 = readbin_vector(['./',id,'/b1_reconstructed.bin']);
        b1 = permute(reshape(b1,[m,n,ncoils]),[2 1 3]);
    end

    if exist(['./',id,'/u0_reconstructed.bin'])==2
        u0 = readbin_vector(['./',id,'/u0_reconstructed.bin']);
        u0 = permute(reshape(u0,[m,n]),[2 1]);
    end

    if exist(['./',id,'/datanorm_factor.bin'])==2
        datanorm = readbin_vector(['./',id,'/datanorm_factor.bin']);
    end


    if prescale;
        datanorm = datanorm*datanorm_pre;
    end
    
  
    % clean up
    unix(['rm -rf ./',id])

end
