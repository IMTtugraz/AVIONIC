function [croppedData,cropIndices] = removeOS(data, dim, percent, center)

N = size(data,dim);
p = 1:ndims(data);
p([1 dim]) = p([dim 1]);

if dim~=1
data = permute(data,p);
end

if (nargin < 3)
    percent=2;
end

if (nargin < 4)
    center = N/2;
end

% compensate asymmetric echo
sh = zeros(ndims(data),1);
shift = N-2*(center);
if shift~=0
    sh(1) = shift;
    data = padarray(data, sh,'pre');
end

% compute new N
N = size(data, 1);

% remove os
diff = N-floor(N/percent);
cropIndices = floor(diff/2)+1:floor(N/percent)+floor(diff/2);
%cropIndices = floor(N/4)+1:3*floor(N/4);

imgOS = sqrt(N) * ifftshift(ifft(fftshift(data,1),[],1),1);
% dirty work-around
str = ',:';str = repmat(str,[1 ndims(data)-1]);str = cat(2,'cropIndices',str);
%eval(['imgOS(',str,')=[];']);
eval(['imgOS=imgOS(',str,');']);

n = size(imgOS,1);
croppedData = 1./sqrt(N) * fftshift(fft(ifftshift(imgOS,1),[],1),1);
if dim~=1
    croppedData = permute(croppedData,p);
end
                                             
