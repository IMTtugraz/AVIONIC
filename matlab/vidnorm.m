function [y] = vidnorm(x);

y = uint8( (255/max(x(:))) * x);
