function [errorv, errorvroi] = myerror(x,y,method,roi,subtract_mean)
% Calculate different error metrics
%
% INPUT
% x:  Reconstruction
% y:  Reference Image (ideal)
% roi: region of interest
% method: mse, rmse, nmse, nrmse, ser
%
% OUTPUT
% errorv:
% errorvroi:
%--------------------------------------------------------------------------

 x = abs(x);
 y = abs(y);

[n,m,nframes] = size(x);

if nargin<5
    subtract_mean=0;
end

if nargin<4
    roi = (ones(size(x)));
end

if numel(roi) ~= numel(x)
    error('ROI dimensions do not fit image dimensions');
end

[n,m,t] = size(x);

x = double(x(:));
y = double(y(:));

if ~islogical(roi)
    roi = logical(roi);
end

xroi = x(roi);
yroi = y(roi);


x = x(:);
y = y(:);

if subtract_mean
    xroi = xroi - sum(xroi)./sum(roi(:));
    yroi = yroi - sum(yroi)./sum(roi(:));
    
    % subtract mean
    x = x(:) - (sum(sum(sum(x))))./(n*m*t);
    y = y(:) - (sum(sum(sum(y))))./(n*m*t);
end

switch method
    
    case 'mse' % mean-squared-error
        errorv = sum( (x-y).^2 );
        errorvroi = sum( (xroi - yroi).^2 );
    case 'rmse' % root-mean-squared-error
        errorv = sqrt( sum( (x-y).^2 ) );
        errorvroi = sqrt( sum( (xroi - yroi).^2 ) );
    case 'nmse' % normalized-root-mean-squared-error
        errorv = ( sum( (x-y).^2 ) )./ sum(y.^2);
        errorvroi = ( sum( (xroi - yroi).^2 ) ) ./ sum(y.^2);
    case 'nrmse'% normalized-root-mean-squared-error
        errorv = sqrt( sum( (x-y).^2 ) )./ sqrt(sum(y.^2));
        errorvroi = sqrt( sum( (xroi - yroi).^2 ) ) ./ sqrt(sum(y.^2));
    case 'ser' % signal-to-error-ratio (dB)
        errorv = -10*log10( (sum((x-y).^2))./(sum(y.^2)) );
        errorvroi = -10*log10( (sum((xroi - yroi).^2))./(sum(yroi.^2)) );
    case 'serfro'
        x=reshape(x,[n*m, nframes]);
        y=reshape(y,[n*m, nframes]);
        roi = reshape(roi,[n*m,nframes]);
        errorv = -10*log10( (norm((x-y),'fro')^2)/(norm(y,'fro')^2) );
        errorvroi = -10*log10( (norm((x-y).*roi,'fro')^2)/(norm(y.*roi,'fro')^2) );
    case 'serktrpca'
        errorv = -10*log10( (norm( reshape(x,[n*m nframes])-reshape(y,[n*m nframes]),'fro')^2)/(norm(reshape(y,[n*m nframes]),'fro')^2) );
        errorvroi = 0;
    case 'psnr'
        errorv    = psnr(x,y,max(abs(y(:))));
        errorvroi = psnr(xroi,yroi,max(abs(y(:))));
    otherwise
        error('no valid metric')
        

        
end
