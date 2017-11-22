function gradt = fgrad_t(u)


    gradt = zeros(size(u));

    gradt(:,:,1:end-1) = u(:,:,2:end) - u(:,:,1:end-1);

end

