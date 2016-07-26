%Function to compute the weight of the spatial derivative (w1) and the weight of the temporal derivative (w2) from a given ratio
%The weights are computed such that the normalized integral over all possible directions equals 1
%The ratio behind this is to make the ratio a true "model" parameter and avoid an implicit change of the cost of the regularization functional
function [w1,w2] = get_weights_3d(ratio)

%ratio: time/space weight ratio: ratio = w2/w1
%w1: space weight, w2: time weight



%Ass the ellipke function accepts only positiv arguments, we replace ratio by 1/ratio in case ratio > 1
ratio = 1/ratio;
if (0<ratio) && (ratio<=1)
    

    [~,tmp] = ellipke(1-ratio^2);
    w2 = pi/(2*tmp);
    w1 = ratio*w2;
    
elseif (ratio>1)
    
    ratio = 1/ratio;
    [~,tmp] = ellipke(1-ratio^2);
    w1 = pi/(2*tmp);
    w2 = ratio*w1;

else
    error('Wrong ratio!');
end


