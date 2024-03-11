% Input database files into Matlab
       clear;
       no_of_images=10;
% Reading images from the database. The image files should be located in the subfolder ‘ database"
       for j=1:no_of_images
           imported_image=imread(['database/s1/' num2str(j) '.pgm']);
           [m,n]=size(imported_image);
           B(:,j)=reshape(imported_image,m*n,1);
       end;

% % clear;
% % ATTFolder = "database_full/";
% % 
% % folders = dir(fullfile(ATTFolder, 's*'));
% % NL = numel(folders);
% % no_of_images=400;
% % 
% % for i=1:NL
% %     files = dir(fullfile(ATTFolder, folders(i).name, '*.pgm'))
% %     for j=1:numel(files)
% %         fname = fullfile(ATTFolder, folders(i).name, files(j).name);
% %         img = imread(fname);
% %         [m,n]=size(img);
% %         B(:,i*j)=reshape(img,m*n,1);
% %     end
% % end

% Computing the mean face
     mean_face=mean(B,2);

% Displaying the mean face
     imshow(uint8(reshape(mean_face,m,n)))

% Subtract the mean face
     B=double(B);
     B=B-mean_face;

% Compute the covariance matrix of the set 

    covar_matrix=B'*B;

% Compute its eigenvalues and Eigen_Vectors

     [Vectors,Values]=eig(covar_matrix);
     Eigen_Vectors=B*Vectors;

% Eigen_weights
    Eigen_weights = Eigen_Vectors'*B;

% Display the set of eigenfaces
     for j=1:no_of_images;
         if j==1
             Eigen_Faces=reshape(Eigen_Vectors(:,j)+mean_face,m,n);
         else
             Eigen_Faces=[Eigen_Faces reshape(Eigen_Vectors(:,j)+mean_face,m,n)];
         end; 
     end
     Eigen_Faces=uint8(Eigen_Faces);
     figure;
     imshow(Eigen_Faces);

     Products=Eigen_Vectors'*Eigen_Vectors;

% Recognition of digitally altered image (sunglasses)
     image_read1=imread(['1_paint.pbm']);
     U1=reshape(image_read1,m*n,1);
     Norms_Eigen_Vectors1=diag(Products);
     Weight_1=(Eigen_Vectors'*(double(U1)-mean_face));
     W1=(Eigen_Vectors'*(double(U1)-mean_face));

     D=zeros(no_of_images,1);
     for k=1:no_of_images;
        D(k) = norm(Weight_1 - Eigen_weights(:,k));
     end

     D=D';

     [M, I]=min(D);
     
     result_image=imread(['database/s1/' num2str(I) '.pgm']);
     imshow([image_read1,result_image])
%      
% 
% 
% %      Weight_1=Weight_1./Norms_Eigen_Vectors1;
% %      U_approx1=Eigen_Vectors*Weight_1+mean_face;
% %      image_approx1=uint8(reshape(U_approx1,m,n));
% %      figure;
% %      imshow([image_read1,image_approx1]);
        

