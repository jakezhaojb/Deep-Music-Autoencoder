function flag = saveMNISTtotxt(filename)
% save MNIST images to a .txt file

if exist(filename, "file")
    flag = true;
    return;
end

images = loadMNISTImages('train-images-idx3-ubyte');
fp = fopen(filename, "w");
[ro, cn] = size(images);
for i = 1:cn
    disp(["processing the ", int2str(i), "-th image"]);
    fflush(stdout);
    for j = 1:ro
        fprintf(fp, "%f ", images(j, i));
    end
    fprintf(fp, "\n");
end
fclose(fp);

flag = true;

