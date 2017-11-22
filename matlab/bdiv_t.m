
function divt = bdiv_t(v)

    divt = zeros(size(v));

    divt(:,:,1)         = v(:,:,1);
    divt(:,:,2:end)     = v(:,:,2:end) - v(:,:,1:end-1);
    divt(:,:,end)       = -v(:,:,end-1);
end