function [error] = erroscil(measure, scale)
%given the scale and the value it finds the error
for i=1:length(measure);
    error(i)=sqrt((0.04*scale)^2+ (0.01*measure(i)^2));
end

end