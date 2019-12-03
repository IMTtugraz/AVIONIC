function pattern = caipirinha(dimensions,acblock,acc,shift)


    rNx = dimensions(1);
    rNy = dimensions(2);
    rNz = dimensions(3);


  if nargin<4
    if acc>2;shift = acc;else shift=floor(acc/2);end
  end
      
    if (mod(acc,2) == 0) && (acc>3)

        pattern = zeros(rNy,acc/2+mod(acc/2,2));
        pattern(1:shift:end,1) = 1;
        pattern(1+2:shift:end,ceil(acc/4)+1) = 1;
        pattern = repmat(pattern,[1 rNz/(acc/2+mod(acc/2,2))]);


    else
        pattern = zeros(rNy*rNz,1);
        pattern(1:acc:end) = 1;
        pattern = reshape( pattern,[rNy,rNz] );
    end

    % autocalibration block
    pattern(floor( (rNy-acblock(1))/2) + 1 : floor( (rNy+acblock(1))/2) ,...
        floor( (rNz-acblock(2))/2) + 1:floor( (rNz+acblock(2))/2)  ) = 1;
    pattern = repmat(pattern,[1 1 rNx]);
    pattern = permute(pattern,[3 1 2]);

    
end

