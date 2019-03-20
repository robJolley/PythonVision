bw = imread('11Errosions.bmp');
bw =imresize(bw,0.25);
col1 = imread('resize.bmp');
imshow(bw)
s = regionprops(bw,{...
    'Centroid',...
    'MajorAxisLength',...
    'MinorAxisLength',...
    'Orientation'});
figure;
imshow(col1,'InitialMagnification','fit');
t = linspace(0,2*pi,50);

hold on
for k = 1:length(s)
    a = (s(k).MajorAxisLength/2)+11;
    b = (s(k).MinorAxisLength/2)+11;
    Xc = s(k).Centroid(1);
    Yc = s(k).Centroid(2);
    phi = deg2rad(-s(k).Orientation);
    x = Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
    y = Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);
    plot(x,y,'r','Linewidth',1)
end
hold off