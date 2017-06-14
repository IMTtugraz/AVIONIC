function [mri_obj] = setup_data3d(mri_obj)

    [n,m,l,ncoils] = size(mri_obj.data);


    %Get chop variable
    [x,y,z] = meshgrid(1:m,1:n,1:l);
    chop = (-1).^(x+y+z);
    clear x y z

    %Shift mask and data for matlab fft , recenter image by chop
    mri_obj.mask = ifftshift( mri_obj.mask );


    for coil=1:ncoils
        mri_obj.data(:,:,:,coil) = ifftshift(chop.*mri_obj.data(:,:,:,coil));
    end

    %Rescale data
    mri_obj.data = mri_obj.data*sqrt(n*m*l);

end
