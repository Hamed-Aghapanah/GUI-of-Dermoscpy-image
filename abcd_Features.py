def abcd_Features(img): #image == mask
    import numpy as np
    import cv2
#    img=np.uint16(img)    
    show=0
    show_entropy=1;
    show_gabor=11;
    
    texture_features=0
    TETHA_GABOR =[0,1]#,2,3]
    Frequency_GABOR=[0.1,0.2]#,0.3, 0.4,0.5]
    if show==1:
        print('------------------------------------------')
        print('Segmentation is done by Hamed Aghapanah ')
        print('------------------------------------------')

    #from PIL import ImageOps
    #import matplotlib as mpl
    from skimage.draw import ellipse as ellipse1
    #img = cv2.imread('image\skin (3).bmp')
#    img = cv2.imread('image/skin.bmp', -1)
    
    if show==1:
        #import matplotlib
        #import matplotlib.cm as cm
        #import matplotlib.mlab as mlab
        #import matplotlib.pyplot as plt
        #img = cv2.imread('image/abcde.jpg')
        #cv2.imshow('abcde',img)
        # Read the image and perfrom an OTSU threshold
        
        img = cv2.imread('image\skin (3).bmp')
        #img = cv2.imread('image\image3.jpg')
        #img = cv2.imread('image\skin (1).jpg')
        img = cv2.imread('image\skin (2).jpg')
        #img = cv2.imread('image\skin (3).jpg');  ssssss=0 
        #img = cv2.imread('image\skin (1).bmp')
        #img = cv2.imread('image\skin (2).bmp')
        
        #img = cv2.imread('image\imagea.PNG')
        #img = cv2.imread('image\imageb.PNG')
        #img = cv2.imread('image\imagec.PNG')
        img = cv2.imread('image/skin.bmp', -1)
    
#        cv2.imshow('sss',img)
#        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    image2=img.copy()
    image2=img
    
    features=[]
    image0=img[:]
    img2=img.copy()
    #skin (3).bmp
    #img= cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    number_element = np.nanargmax(img);#img = cv2.resize(img, (222, 248))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh =     cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Remove hair with opening
    kernel = np.ones((10,10),np.uint8)
    opening1 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    opening = cv2.morphologyEx(opening1,cv2.MORPH_CLOSE,kernel, iterations = 2)
    # Combine surrounding noise with ROI
    kernel = np.ones((2,2),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    # Blur the image for smoother ROI
    blur = cv2.blur(dilate,(15,15))
    # Perform another OTSU threshold and search for biggest contour
    ret, thresh =cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    if show==1:
#        cv2.imshow('s',thresh)
#        cv2.imshow('img',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    #img = cv2.imread('skin.jpg')
    #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(imgray,127,255,0)
    #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #s,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    #ssss = contours.shape
    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    len_contours = len(contours)
    
    #cv2.imshow('res', res)
    #if show==1:
    #    cv2.imshow('mask', mask)
    # Display the result
    #cv2.imwrite(IMD+'.png', res)
    ss=[]
    for i in range(len_contours):
        ccc=contours[i]
        ss=np.append(ss,len(ccc))
       
    ss_max = np.max(ss)   
    countour_cnt =np.where( ss >= ss_max )
    if show==1:
        print(countour_cnt)
    #print('ss_max',ss_max)
    #countour_cnt=  
    try:
        x=np.int(countour_cnt[0])
    except:x=0    
    if show==1:
        print(x)
    xx=1;xx=x
    cnt1 = contours[xx]
    cnt = contours[xx]
    
    # =============================================================================
    # features
    # =============================================================================
    if show==1:
        print('--------------------------------------------------------------')
        print('ABCD features')
        print('--------------------------------------------------------------')
    
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    #hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
    hull = cv2.convexHull(cnt)
    #7. Bounding Rectangle
    #There are two types of bounding rectangles.
    #7.a. Straight Bounding Rectangle
    #It is a straight rectangle, it doesn't consider the rotation of the object. So area of the bounding rectangle won't be minimum. It is found by the function cv2.boundingRect().
    #
    #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #7.b. Rotated Rectangle
    #Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. The function used is cv2.minAreaRect(). It returns a Box2D structure which contains following detals - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv2.boxPoints()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if show==1:
        cv2.drawContours(img,[box],0,(0,0,255),2)
    #Both the rectangles are shown in a single image. Green rectangle shows the normal bounding rect. Red rectangle is the rotated rect.
    #boundingrect.png
    #image
    #8. Minimum Enclosing Circle
    #Next we find the circumcircle of an object using the function cv2.minEnclosingCircle(). It is a circle which completely covers the object with minimum area.
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    #cv2.circle(img,center,radius,(0,255,0),2)
    #circumcircle.png
    #image
    #9. Fitting an Ellipse
    #Next one is to fit an ellipse to an object. It returns the rotated rectangle in which the ellipse is inscribed.
    ellipse = cv2.fitEllipse(cnt)
    (x1,y1),(MA,ma),angle= cv2.fitEllipse(cnt)
    if show==1:
        cv2.ellipse(img,ellipse,(0,255,255),2)
    #fitellipse.png
    #image
    #10. Fitting a Line
    #Similarly we can fit a line to a set of points. Below image contains a set of white points. We can approximate a straight line to it.
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    
    if show==1:
        cv2.line(img,(cols-1,righty),(0,lefty),(200,255,60),2)
    
    Asymmetrical1=np.fix(10000*area/(radius*number_element))
    Asymmetrical2=np.fix(10000*area/(epsilon*number_element))
    
    h, w = img.shape[:2] 
    ellipse0= np.zeros((h, w), dtype=np.uint8)
    rr, cc = ellipse1( y1,x1, MA/2, ma/2, rotation=np.deg2rad(90-angle))
    print('rr',rr,'cc',cc)
    if rr[0]>=1 and cc[0]>=1:
        ellipse0[rr, cc] = 1
        #cv2.imshow('ellipse0',ellipse0)
        #skimage.draw.ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.0)
        inside =(np.max(mask)-mask)*ellipse0
        #cv2.imshow('inside',inside)
        outside= mask*(1-ellipse0)
    else:
        inside=10
        outside=30
    #cv2.imshow('outside',outside)
    Border1=(outside+inside)
    
    #if show==1:
    #    cv2.imshow('Border1',Border1)
    Border=np.fix(1000*np.mean(Border1)/np.mean(mask))/1000
    Color1=0.85*np.fix(100000*np.mean(img2[:,:,0])/(np.mean(img2[:,:,1]) *np.mean(img2[:,:,2])))/1000
    Diameter1=np.fix(1000*radius/np.max([h,w]))/300
    #cv2.imshow('s2',img)
    
    
    Diameter=Diameter1*(3)
    if show==1:
    
        print ('Asym=',Asymmetrical1,'(index), Asym2=',Asymmetrical2,'(index)')
        print ('Border=',Border,'(index)')
        print ('Color=',Color1)
        print ('Diameter=',Diameter,' cm')
    
    features.append((Asymmetrical1/15))
    features.append((Border))
    features.append((Color1))
    features.append((Diameter))
    
    #print ('Evolution =',Evolution )
    
    
    x0=box[0,0];y0=box[0,1]
    x1=box[1,0];y1=box[1,1]
    x2=box[2,0];y2=box[2,1]
    x3=box[3,0];y3=box[3,1]
    #print('box = ',box)
    x00=np.min([x0,x1,x2,x3])
    y00=np.min([y0,y1,y2,y3])
    x11=np.max([x0,x1,x2,x3])
    y11=np.max([y0,y1,y2,y3])
    #temp[y00:y11,x00:x11,:]=0
    temp=img2[y00:y11,x00:x11,:]
    if show==1:
        cv2.imshow('coped image',temp)
        
        for i in range(1):
        #while (1):   
            cv2.drawContours(img,[cnt1],-1,(0,255,255),1),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(255,0,255),2),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(255,255,0),3),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(255,0,0),4),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(0,0,255),5),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(0,255,0),0),cv2.imshow("contour", img),cv2.waitKey(100)
#            k = cv2.waitKey(1) #& 0xFF
    
    if show==1:
        print('score : ')
    croped_image=temp.copy()
    
    
    
    # =============================================================================
    # GLCM Features
    # =============================================================================
    fea_glcm=[]
    temp=croped_image.copy()
    #import matplotlib.pyplot as plt
    if show==1:
        print('--------------------------------------------------------------')
        print('GLCM features')
        print('--------------------------------------------------------------')
    from skimage.feature import greycomatrix, greycoprops
    #from skimage import data
    PATCH_SIZE = 21
    patch=temp[:,:,0]
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    g1=(glcm[:,:,0,0])
    #cv2.imshow('g1',g1)
    dissimilarity=(greycoprops(glcm, 'dissimilarity')[0, 0])
    dissimilarity=np.fix(dissimilarity*100)/100
    if show==1:
        print('dissimilarity=',dissimilarity)
    fea_glcm.append((dissimilarity))
    features.append((dissimilarity))
    
    correlation=(greycoprops(glcm, 'correlation')[0, 0])
    correlation=np.fix(correlation*100)/100
    if show==1:
        print('correlation=',correlation)
    fea_glcm.append((correlation))
    features.append((correlation))
    
    energy=(greycoprops(glcm, 'energy')[0, 0])
    energy=np.fix(energy*100)/100
    if show==1:
        print('energy=',energy); 
    fea_glcm.append((energy))
    features.append((energy))
    
    
    homogeneity=(greycoprops(glcm, 'homogeneity')[0, 0])
    homogeneity=np.fix(homogeneity*100)/100
    if show==1:
        print('homogeneity=',homogeneity)
    fea_glcm.append((homogeneity))
    features.append((homogeneity))
    
    
    ASM=(greycoprops(glcm, 'ASM')[0, 0])
    ASM=np.fix(ASM*100)/100
    if show==1:
        print('ASM=',ASM)
    fea_glcm.append((ASM))
    features.append((ASM))
    
    
    contrast=(greycoprops(glcm, 'contrast')[0, 0])
    contrast=np.fix(contrast*100)/100
    if show==1:
        print('contrast=',contrast)
    features.append((contrast))
    
    fea_glcm.append((contrast))
    #fea_glcm = np.ravel(fea_glcm)
    #fea_glcm=np.transpose(fea_glcm)
    #features.append((fea_glcm))
    if show==1:
        print('fea_glcm',fea_glcm)
    
        print('')
        print('')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #shade=(greycoprops(glcm, 'shade')[0, 0])
    #print('shade=',shade)
    # prominence=(greycoprops(glcm, 'prominence')[0, 0])
    #print('prominence=',prominence)
    #entropy=(greycoprops(glcm, 'entropy')[0, 0])
    #print('entropy=',entropy)
    #entropy=(greycoprops(glcm)[0, 0]) /normalizeh_entropy
    #print('entropy=',entropy)
    #features.append((entropy))
    
    # =============================================================================
    # GLRLM Features
    # ============================================================================
    #glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    
    
    # =============================================================================
    # shape index Features
    # =============================================================================
    #print('--------------------------------------------------------------')
    #print('shape index features')
    #print('--------------------------------------------------------------')
    # 
    #from skimage.feature import shape_index
    #square = np.zeros((5, 5))
    #square[2, 2] = 4
    #s = shape_index(mask, sigma=0.1)
    ##cv2.imshow('s',s)
    # =============================================================================
    # Gabor Features
    # =============================================================================
    if show==1:
        print('--------------------------------------------------------------')
        print('Gabor features')
        print('--------------------------------------------------------------')
    temp=croped_image.copy() 
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi
    fea_ga=features
    
    from skimage import data
    from skimage.util import img_as_float
    from skimage.filters import gabor_kernel
    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats
    
    
    
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    
    
    shrink = (slice(0, None, 3), slice(0, None, 3))
    brick = img_as_float(data.load('brick.png'))[shrink]
    wall222 = img_as_float((temp))[shrink];
    wall222=wall222[:,:,0]
    
    fx,fy=brick.shape
    #thumbnail = CreateMat(brick.rows,brick.col, cv2.CV_8UC3)
    #    cv.Resize(im, thumbnail)
        
    #t1=cv2.resize(wall222, fx,fy) 
    height, width = brick.shape 
    brick = cv2.resize(wall222,(round(width),round(height)),interpolation = cv2.INTER_CUBIC)
    grass=brick
    wall3=brick
    wall=brick
    #brick3 = cv2.Resize(brick2, brick)
    #brick=brick2.copy()       
    image_names = ('brick')#, 'grass', 'wall')
    images = (brick, grass, wall)
    
    # prepare reference features
    ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
    ref_feats[ :, :] = compute_feats(brick, kernels)
    
    def power(image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    
    # Plot a selection of the filter bank kernels and their responses.
    results = []
    kernel_params = []
    for theta in ( TETHA_GABOR ):
        theta = theta / 4. * np.pi
        for frequency in (Frequency_GABOR):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            
    #        all
            x=(kernel, [power(img, kernel) for img in images]) 
            xx0=x[0]
            xx1=x[1]
    #        print('ssssssssssss',np.abs(np.mean(xx0)))
            fea_ga.append(( np.abs(np.mean(xx0)) ,np.abs(np.mean(xx1)),np.abs(np.var(xx0)) , np.abs(np.var(xx1)) ))
            
            results.append((kernel, [power(img, kernel) for img in images]))
    if show_gabor==1:
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(5, 6))
        plt.gray()
        fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
        axes[0][0].axis('off')
    
        # Plot original images
        for label, img, ax in zip(image_names, images, axes[0][1:]):
            ax.imshow(img)
            ax.set_title(label, fontsize=9)
            ax.axis('off')
        
        for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
            # Plot Gabor kernel
            ax = ax_row[0]
            ax.imshow(np.real(kernel), interpolation='nearest')
            ax.set_ylabel(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        
            # Plot Gabor responses with the contrast normalized for each filter
            vmin = np.min(powers)
            vmax = np.max(powers)
            for patch, ax in zip(powers, ax_row[1:]):
                ax.imshow(patch, vmin=vmin, vmax=vmax)
                ax.axis('off')
        
        plt.show()
    # feats
    a1,a2,a3=ref_feats.shape
    ref_feats = np.reshape(ref_feats,(a2,a1*a3))
    fref_feats = np.ravel(ref_feats)
    #fea_gabor=ref_feats.flatten()
    #print('fref_feats.shape',len(fref_feats))
    for i in range(len(fref_feats)):
        t=fref_feats[i]
        features.append(t)
    features.append((contrast))
    
    return ( features ,img, cnt1)



def abcd_Features1(img):
    show=0
    show_entropy=1;
    show_gabor=11;
    
    texture_features=0
    TETHA_GABOR =[0,1]#,2,3]
    Frequency_GABOR=[0.1,0.2]#,0.3, 0.4,0.5]
    if show==1:
        print('------------------------------------------')
        print('Segmentation is done by Hamed Aghapanah ')
        print('------------------------------------------')
    import numpy as np
    import cv2
    #from PIL import ImageOps
    #import matplotlib as mpl
    from skimage.draw import ellipse as ellipse1
    #img = cv2.imread('image\skin (3).bmp')
#    img = cv2.imread('image/skin.bmp', -1)
    
    if show==1:
        #import matplotlib
        #import matplotlib.cm as cm
        #import matplotlib.mlab as mlab
        #import matplotlib.pyplot as plt
        #img = cv2.imread('image/abcde.jpg')
        #cv2.imshow('abcde',img)
        # Read the image and perfrom an OTSU threshold
        
        img = cv2.imread('image\skin (3).bmp')
        #img = cv2.imread('image\image3.jpg')
        #img = cv2.imread('image\skin (1).jpg')
        img = cv2.imread('image\skin (2).jpg')
        #img = cv2.imread('image\skin (3).jpg');  ssssss=0 
        #img = cv2.imread('image\skin (1).bmp')
        #img = cv2.imread('image\skin (2).bmp')
        
        #img = cv2.imread('image\imagea.PNG')
        #img = cv2.imread('image\imageb.PNG')
        #img = cv2.imread('image\imagec.PNG')
        img = cv2.imread('image/skin.bmp', -1)
    
#        cv2.imshow('sss',img)
#        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    image2=img.copy()
    
    
    features=[]
    image0=img[:]
    img2=img.copy()
    #skin (3).bmp
    #img= cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    number_element = np.nanargmax(img);#img = cv2.resize(img, (222, 248))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh =     cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Remove hair with opening
    kernel = np.ones((10,10),np.uint8)
    opening1 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    opening = cv2.morphologyEx(opening1,cv2.MORPH_CLOSE,kernel, iterations = 2)
    # Combine surrounding noise with ROI
    kernel = np.ones((2,2),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    # Blur the image for smoother ROI
    blur = cv2.blur(dilate,(15,15))
    # Perform another OTSU threshold and search for biggest contour
    ret, thresh =cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    if show==1:
#        cv2.imshow('s',thresh)
#        cv2.imshow('img',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    #img = cv2.imread('skin.jpg')
    #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(imgray,127,255,0)
    #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #s,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    #ssss = contours.shape
    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    len_contours = len(contours)
    
    #cv2.imshow('res', res)
    #if show==1:
    #    cv2.imshow('mask', mask)
    # Display the result
    #cv2.imwrite(IMD+'.png', res)
    ss=[]
    for i in range(len_contours):
        ccc=contours[i]
        ss=np.append(ss,len(ccc))
       
    ss_max = np.max(ss)   
    countour_cnt =np.where( ss >= ss_max )
    if show==1:
        print(countour_cnt)
    #print('ss_max',ss_max)
    #countour_cnt=    
    x=np.int(countour_cnt[0])
    if show==1:
        print(x)
    xx=1;xx=x
    cnt1 = contours[xx]
    cnt = contours[xx]
    
    # =============================================================================
    # features
    # =============================================================================
    if show==1:
        print('--------------------------------------------------------------')
        print('ABCD features')
        print('--------------------------------------------------------------')
    
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    #hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
    hull = cv2.convexHull(cnt)
    #7. Bounding Rectangle
    #There are two types of bounding rectangles.
    #7.a. Straight Bounding Rectangle
    #It is a straight rectangle, it doesn't consider the rotation of the object. So area of the bounding rectangle won't be minimum. It is found by the function cv2.boundingRect().
    #
    #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #7.b. Rotated Rectangle
    #Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. The function used is cv2.minAreaRect(). It returns a Box2D structure which contains following detals - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv2.boxPoints()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if show==1:
        cv2.drawContours(img,[box],0,(0,0,255),2)
    #Both the rectangles are shown in a single image. Green rectangle shows the normal bounding rect. Red rectangle is the rotated rect.
    #boundingrect.png
    #image
    #8. Minimum Enclosing Circle
    #Next we find the circumcircle of an object using the function cv2.minEnclosingCircle(). It is a circle which completely covers the object with minimum area.
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    #cv2.circle(img,center,radius,(0,255,0),2)
    #circumcircle.png
    #image
    #9. Fitting an Ellipse
    #Next one is to fit an ellipse to an object. It returns the rotated rectangle in which the ellipse is inscribed.
    ellipse = cv2.fitEllipse(cnt)
    (x1,y1),(MA,ma),angle= cv2.fitEllipse(cnt)
    if show==1:
        cv2.ellipse(img,ellipse,(0,255,255),2)
    #fitellipse.png
    #image
    #10. Fitting a Line
    #Similarly we can fit a line to a set of points. Below image contains a set of white points. We can approximate a straight line to it.
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    
    if show==1:
        cv2.line(img,(cols-1,righty),(0,lefty),(200,255,60),2)
    
    Asymmetrical1=np.fix(10000*area/(radius*number_element))
    Asymmetrical2=np.fix(10000*area/(epsilon*number_element))
    
    h, w = img.shape[:2] 
    ellipse0= np.zeros((h, w), dtype=np.uint8)
    rr, cc = ellipse1( y1,x1, MA/2, ma/2, rotation=np.deg2rad(90-angle))
    ellipse0[rr, cc] = 1
    #cv2.imshow('ellipse0',ellipse0)
    #skimage.draw.ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.0)
    inside =(np.max(mask)-mask)*ellipse0
    #cv2.imshow('inside',inside)
    outside= mask*(1-ellipse0)
    #cv2.imshow('outside',outside)
    Border1=(outside+inside)
    
    #if show==1:
    #    cv2.imshow('Border1',Border1)
    Border=np.fix(1000*np.mean(Border1)/np.mean(mask))/1000
    Color1=0.85*np.fix(100000*np.mean(img2[:,:,0])/(np.mean(img2[:,:,1]) *np.mean(img2[:,:,2])))/1000
    Diameter1=np.fix(1000*radius/np.max([h,w]))/300
    #cv2.imshow('s2',img)
    
    
    Diameter=Diameter1*(3)
    if show==1:
    
        print ('Asym=',Asymmetrical1,'(index), Asym2=',Asymmetrical2,'(index)')
        print ('Border=',Border,'(index)')
        print ('Color=',Color1)
        print ('Diameter=',Diameter,' cm')
    
    features.append((Asymmetrical1/15))
    features.append((Border))
    features.append((Color1))
    features.append((Diameter))
    
    #print ('Evolution =',Evolution )
    
    
    x0=box[0,0];y0=box[0,1]
    x1=box[1,0];y1=box[1,1]
    x2=box[2,0];y2=box[2,1]
    x3=box[3,0];y3=box[3,1]
    #print('box = ',box)
    x00=np.min([x0,x1,x2,x3])
    y00=np.min([y0,y1,y2,y3])
    x11=np.max([x0,x1,x2,x3])
    y11=np.max([y0,y1,y2,y3])
    #temp[y00:y11,x00:x11,:]=0
    temp=img2[y00:y11,x00:x11,:]
    if show==1:
        cv2.imshow('coped image',temp)
        
        for i in range(1):
        #while (1):   
            cv2.drawContours(img,[cnt1],-1,(0,255,255),1),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(255,0,255),2),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(255,255,0),3),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(255,0,0),4),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(0,0,255),5),cv2.imshow("contour", img),cv2.waitKey(100)
            cv2.drawContours(img,[cnt1],-1,(0,255,0),0),cv2.imshow("contour", img),cv2.waitKey(100)
#            k = cv2.waitKey(1) #& 0xFF
    
    if show==1:
        print('score : ')
    croped_image=temp.copy()
    
    
    
    # =============================================================================
    # GLCM Features
    # =============================================================================
    fea_glcm=[]
    temp=croped_image.copy()
    #import matplotlib.pyplot as plt
    if show==1:
        print('--------------------------------------------------------------')
        print('GLCM features')
        print('--------------------------------------------------------------')
    from skimage.feature import greycomatrix, greycoprops
    #from skimage import data
    PATCH_SIZE = 21
    patch=temp[:,:,0]
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    g1=(glcm[:,:,0,0])
    #cv2.imshow('g1',g1)
    dissimilarity=(greycoprops(glcm, 'dissimilarity')[0, 0])
    dissimilarity=np.fix(dissimilarity*100)/100
    if show==1:
        print('dissimilarity=',dissimilarity)
    fea_glcm.append((dissimilarity))
    features.append((dissimilarity))
    
    correlation=(greycoprops(glcm, 'correlation')[0, 0])
    correlation=np.fix(correlation*100)/100
    if show==1:
        print('correlation=',correlation)
    fea_glcm.append((correlation))
    features.append((correlation))
    
    energy=(greycoprops(glcm, 'energy')[0, 0])
    energy=np.fix(energy*100)/100
    if show==1:
        print('energy=',energy); 
    fea_glcm.append((energy))
    features.append((energy))
    
    
    homogeneity=(greycoprops(glcm, 'homogeneity')[0, 0])
    homogeneity=np.fix(homogeneity*100)/100
    if show==1:
        print('homogeneity=',homogeneity)
    fea_glcm.append((homogeneity))
    features.append((homogeneity))
    
    
    ASM=(greycoprops(glcm, 'ASM')[0, 0])
    ASM=np.fix(ASM*100)/100
    if show==1:
        print('ASM=',ASM)
    fea_glcm.append((ASM))
    features.append((ASM))
    
    
    contrast=(greycoprops(glcm, 'contrast')[0, 0])
    contrast=np.fix(contrast*100)/100
    if show==1:
        print('contrast=',contrast)
    features.append((contrast))
    
    fea_glcm.append((contrast))
    #fea_glcm = np.ravel(fea_glcm)
    #fea_glcm=np.transpose(fea_glcm)
    #features.append((fea_glcm))
    if show==1:
        print('fea_glcm',fea_glcm)
    
        print('')
        print('')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #shade=(greycoprops(glcm, 'shade')[0, 0])
    #print('shade=',shade)
    # prominence=(greycoprops(glcm, 'prominence')[0, 0])
    #print('prominence=',prominence)
    #entropy=(greycoprops(glcm, 'entropy')[0, 0])
    #print('entropy=',entropy)
    #entropy=(greycoprops(glcm)[0, 0]) /normalizeh_entropy
    #print('entropy=',entropy)
    #features.append((entropy))
    
    # =============================================================================
    # GLRLM Features
    # ============================================================================
    #glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    
    
    # =============================================================================
    # shape index Features
    # =============================================================================
    #print('--------------------------------------------------------------')
    #print('shape index features')
    #print('--------------------------------------------------------------')
    # 
    #from skimage.feature import shape_index
    #square = np.zeros((5, 5))
    #square[2, 2] = 4
    #s = shape_index(mask, sigma=0.1)
    ##cv2.imshow('s',s)
    # =============================================================================
    # Gabor Features
    # =============================================================================
    if show==1:
        print('--------------------------------------------------------------')
        print('Gabor features')
        print('--------------------------------------------------------------')
    temp=croped_image.copy() 
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi
    fea_ga=features
    
    from skimage import data
    from skimage.util import img_as_float
    from skimage.filters import gabor_kernel
    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats
    
    
    
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    
    
    shrink = (slice(0, None, 3), slice(0, None, 3))
    brick = img_as_float(data.load('brick.png'))[shrink]
    wall222 = img_as_float((temp))[shrink];
    wall222=wall222[:,:,0]
    
    fx,fy=brick.shape
    #thumbnail = CreateMat(brick.rows,brick.col, cv2.CV_8UC3)
    #    cv.Resize(im, thumbnail)
        
    #t1=cv2.resize(wall222, fx,fy) 
    height, width = brick.shape 
    brick = cv2.resize(wall222,(round(width),round(height)),interpolation = cv2.INTER_CUBIC)
    grass=brick
    wall3=brick
    wall=brick
    #brick3 = cv2.Resize(brick2, brick)
    #brick=brick2.copy()       
    image_names = ('brick')#, 'grass', 'wall')
    images = (brick, grass, wall)
    
    # prepare reference features
    ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
    ref_feats[ :, :] = compute_feats(brick, kernels)
    
    def power(image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    
    # Plot a selection of the filter bank kernels and their responses.
    results = []
    kernel_params = []
    for theta in ( TETHA_GABOR ):
        theta = theta / 4. * np.pi
        for frequency in (Frequency_GABOR):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            
    #        all
            x=(kernel, [power(img, kernel) for img in images]) 
            xx0=x[0]
            xx1=x[1]
    #        print('ssssssssssss',np.abs(np.mean(xx0)))
            fea_ga.append(( np.abs(np.mean(xx0)) ,np.abs(np.mean(xx1)),np.abs(np.var(xx0)) , np.abs(np.var(xx1)) ))
            
            results.append((kernel, [power(img, kernel) for img in images]))
    if show_gabor==1:
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(5, 6))
        plt.gray()
        fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
        axes[0][0].axis('off')
    
        # Plot original images
        for label, img, ax in zip(image_names, images, axes[0][1:]):
            ax.imshow(img)
            ax.set_title(label, fontsize=9)
            ax.axis('off')
        
        for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
            # Plot Gabor kernel
            ax = ax_row[0]
            ax.imshow(np.real(kernel), interpolation='nearest')
            ax.set_ylabel(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        
            # Plot Gabor responses with the contrast normalized for each filter
            vmin = np.min(powers)
            vmax = np.max(powers)
            for patch, ax in zip(powers, ax_row[1:]):
                ax.imshow(patch, vmin=vmin, vmax=vmax)
                ax.axis('off')
        
        plt.show()
    # feats
    a1,a2,a3=ref_feats.shape
    ref_feats = np.reshape(ref_feats,(a2,a1*a3))
    fref_feats = np.ravel(ref_feats)
    #fea_gabor=ref_feats.flatten()
    #print('fref_feats.shape',len(fref_feats))
    for i in range(len(fref_feats)):
        t=fref_feats[i]
        features.append(t)
    features.append((contrast))
    
    return ( features ,img, cnt1)