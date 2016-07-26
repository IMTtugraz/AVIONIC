function [ ] = writebin_vector( v, filename )
%WRITEBIN_VECTOR saves a vector in binary format
%   If filename is not set a save dialog will be opened

    path = '';

    if nargin < 1
        error('Not enough input arguments.');
    elseif nargin == 1
        [file, path] = uiputfile({'*.bin','Binary vector file (*.bin)';'*.*',  'All Files (*.*)'},'Save vector ...');
        filename = fullfile(path, file);
    end
    
    if isequal(filename,0) || isequal(path,0)
        % user pressed cancel
        return;
    end
    
    % try to open a file for writing (binary)
    [fid, message] = fopen(filename, 'w');
    if fid == -1
        error(message);
    end

    % ensure double type
    v = double(v(:));

    % check for complex numbers
    iscomplex = 0;
    if (~isreal(v))
        iscomplex = 1;
    end
    
    % write file header information for complex
    fwrite(fid, iscomplex, 'uchar');
    
    % write number of vector elements
    fwrite(fid, length(v), 'uint32');
    
    % write vector data
    fwrite(fid, real(v), 'double');
    if iscomplex
        fwrite(fid, imag(v), 'double');
    end

    fclose(fid);
    
    disp(sprintf('Vector "%s" saved to: "%s"', inputname(1), filename));
end
