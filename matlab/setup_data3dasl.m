function [data,mask] = setup_data3dasl(data,mask)

    [n,m,l,ncoils,frames] = size(data);


    %Get chop variable
    [x,y,z] = meshgrid(1:m,1:n,1:l);
    chop = (-1).^(x+y+z);
    clear x y z

    for nframes = 1:frames
        %Shift mask and data for matlab fft , recenter image by chop
        for coil=1:ncoils
            mask(:,:,:,coil,nframes) = ifftshift(mask(:,:,:,coil,nframes));
            data(:,:,:,coil,nframes) = ifftshift(chop.*data(:,:,:,coil,nframes));
        end
    end
    %Rescale data
    data = data;%*sqrt(n*m*l);

end
