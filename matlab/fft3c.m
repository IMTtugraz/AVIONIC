function res = fft3c(x)

%x ... k-space data with dimension [Ny, Nx, Nz, Ncoil] 

res = zeros(size(x));

for i = 1:size(x,4) 
    temp = x(:,:,:,i);
    res(:,:,:,i) = 1/sqrt(length(temp(:))).*fftshift(fftn(ifftshift(temp)));
end

