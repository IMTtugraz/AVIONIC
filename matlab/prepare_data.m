function mri_obj = prepare_data(data, par_in)

    %Set parameter#############################################################
    % Standard parameters for artifial subsampling on self-generated datasets


    %Regularization parameters for coil construction
    u_reg = 0.2;
    b1_reg = 2;
    b1_final_reg = 0.1;
    b1_final_nr_it = 1000;
    uH1mu = 1e-5;


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

    mri_obj.data = double(data); clear data;
    [n,m,ncoils,nframes] = size(mri_obj.data);
    mri_obj.mask = (squeeze(abs(mri_obj.data(:,:,1,:)))~=0);


    % recenter data
    mri_obj = setup_data(mri_obj);

    % normalize data
    crec = zeros(n,m,ncoils);
    for j = 1:ncoils
        data = sum( squeeze(mri_obj.data(:,:,j,:)), 3);
        msum = sum(mri_obj.mask,3);
        data(msum>0) = data(msum>0)./msum(msum>0);
        crec(:,:,j) = fft2( data )./sqrt(n*m);
    end
    u = sqrt( sum( abs(crec).^2 , 3) );
    datanorm = 255./median(u(u>=0.9.*max(u(:))));
    clear u crec
    mri_obj.data = mri_obj.data.*datanorm;
    mri_obj.datanorm = datanorm;
    display(['scale data to [0 255] - datanorm = ',num2str(datanorm)]);


    % estimate coil sensitivities
    [mri_obj.b1,mri_obj.u0] = coil_construction_opt(mri_obj.data,mri_obj.mask,...
        {'uH1mu',uH1mu;'u_reg',u_reg;'b1_reg',b1_reg;'b1_final_reg',b1_final_reg;'b1_final_nr_it',b1_final_nr_it});

    mri_obj.u0=mri_obj.u0(:,:,ncoils);

       



