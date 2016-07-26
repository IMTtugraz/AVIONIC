function [mri_obj] = setup_data(mri_obj)

[n,m,ncoils,nframes] = size(mri_obj.data);


%Get chop variable
[x,y] = meshgrid(1:m,1:n);
chop = (-1).^(x+y);	
clear x y

%Shift mask and data for matlab fft , recenter image by chop
for j=1:nframes
	mri_obj.mask(:,:,j) = ifftshift( mri_obj.mask(:,:,j) );
	
	for k=1:ncoils
		mri_obj.data(:,:,k,j) = ifftshift(chop.*mri_obj.data(:,:,k,j));
	end
end


%Rescale data
mri_obj.data = mri_obj.data*sqrt(n*m);
