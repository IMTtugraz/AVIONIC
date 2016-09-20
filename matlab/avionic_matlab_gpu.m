function [ g2, comp1, comp2, b1, u0, pdgap ] = avionic_matlab_gpu( data, par_in, mask, b1_in)
% simple export and import to gpu from matlab

    if nargin < 4
        b1_in = [];
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

    % set zero output
    comp1 = 0; comp2 = 0; pdgap = 0; b1 = 0; u0 = 0;
    
    [n,m,ncoils,nframes] = size(data);
    
    mri_obj.data = data;
    mri_obj.mask = mask;
    clear data mask
    
    % rescale
    mri_obj = setup_data(mri_obj);
    
    % normalize data
    crec = zeros(n,m,ncoils);
    for j = 1:ncoils
        data_ = sum( squeeze(mri_obj.data(:,:,j,:)), 3);
        msum = sum(mri_obj.mask,3);
        data_(msum>0) = data_(msum>0)./msum(msum>0);
        crec(:,:,j) = fft2( data_ )./sqrt(n*m);
    end
    u = sqrt( sum( abs(crec).^2 , 3) );
    datanorm = 255./median(u(u>=0.9.*max(u(:))));
    mri_obj.data = mri_obj.data.*datanorm;      

    % write data to binary file
    writebin_vector(permute(mri_obj.data,[2 1 3 4]),...
        ['./data.bin']);     
    writebin_vector(permute(mri_obj.mask,[2 1 3]),...
        ['./mask.bin']);
    
    if ~isempty(b1_in);   
        u0 = sum(crec.*datanorm.*conj(b1_in),3);   
        writebin_vector(permute(b1_in,[2 1 3 4]),...
            ['./b1.bin']);
        writebin_vector(permute(u0,[2 1]),...
            ['./u0.bin']);
        clear crec u

        recon_cmd=['avionic -i ',num2str(stop_par),...
                    ' --ictgv2.lambda=',num2str(lambda),...
                    ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
                    ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
                    ' --ictgv2.alpha=',num2str(alpha),...
                   ' -m ',method,' -e -a -p ',parfile,' -d ', ...
                    num2str(m),':',num2str(n),':0:',num2str(m),':',num2str(n),':0:',num2str(ncoils), ...
                    ':',num2str(nframes),' -u ./u0.bin -s ./b1.bin ./data.bin ./mask.bin ./result.bin'];
    else
        recon_cmd=['avionic -i ',num2str(stop_par),...
                    ' --ictgv2.lambda=',num2str(lambda),...
                    ' --ictgv2.timeSpaceWeight=',num2str(timeSpaceWeight),...
                    ' --ictgv2.timeSpaceWeight2=',num2str(timeSpaceWeight2),...
                    ' --ictgv2.alpha=',num2str(alpha),...
                   ' -m ',method,' -e -a -p ',parfile,' -d ', ...
                    num2str(m),':',num2str(n),':0:',num2str(m),':',num2str(n),':0:',num2str(ncoils), ...
                    ':',num2str(nframes),' ./data.bin ./mask.bin ./result.bin'];
    end

    display(recon_cmd);
    
    % run reconstruction
    unix(recon_cmd);

    % read results
    g2 = readbin_vector('./result.bin');
    g2 = permute(reshape(g2,[m,n,nframes]),[2 1 3]);

    if exist(['./x3_component'])==2
        comp2 = readbin_vector('./x3_component');
        comp2 = permute(reshape(comp1,[m,n,nframes]),[2 1 3]);
        comp1 = g2-comp2;
    end

    if exist(['./PDGap'])==2
        pdgap = readbin_vector('./PDGap');
        pdgap = abs(pdgap);
    end

    if exist(['./b1_reconstructed.bin'])==2
        b1 = readbin_vector('./b1_reconstructed.bin');
        b1 = permute(reshape(b1,[m,n,ncoils]),[2 1 3]);
    end

    if exist(['./u0_reconstructed.bin'])==2
        u0 = readbin_vector('./u0_reconstructed.bin');
        u0 = permute(reshape(u0,[m,n]),[2 1]);
    end

    g2 = g2./datanorm;
    comp1 = comp1./datanorm;
    comp2 = comp2./datanorm;
    
    % clean up
    unix('rm ./u0_reconstructed.bin ./b1_reconstructed.bin ./PDGap ./result.bin ./x3_component ./data.bin ./mask.bin');

end
