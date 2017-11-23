function res = ifft3c(x)

%x ... image data with dimension [Ny, Nx, Nz, Ncoil] 

res = zeros(size(x));

for i = 1:size(x,4) 
    temp = x(:,:,:,i);
    res(:,:,:,i) = sqrt(length(temp(:))).*ifftshift(ifftn(fftshift(temp)));
end

