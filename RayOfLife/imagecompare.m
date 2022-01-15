function im1 = imagecompare(file1, file2)

  im1 = imread(file1);
  im2 = imread(file2);
  
  imshow(abs(im1 - im2) * 10)
  
end
