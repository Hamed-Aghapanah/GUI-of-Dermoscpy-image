# =============================================================================
# function preprocessing     
# =============================================================================
# =============================================================================
# shadow_removal
# =============================================================================
def shadow_removal (I0):
    import cv2 as cv
    import numpy as np
    
    input_thershold=50
#    input_thershold=th
    th=5
    sigma1=9
    mu1=11
    sigma1=7
    mu1=11
    #img = cv.imread('image1.jpg',1)
    #img = cv.imread('skin (1).bmp',1)
    img0 =I0# cv.imread('image/skin.jpg',1)
    img=img0  
    for i in range(15):
        input_thershold=i*2.5/10
    #    img = cv.imread('skin (1).bmp',1)
        height, width = img.shape[:2]
#        cv.imshow('img',img)  #a, b, c=img.size   #a =np.uint16(a)    #minn=0  #maxx=100
        x, y = np.meshgrid(np.linspace(-(width/2),(width/2),width), np.linspace(-(height/1.4),(height/1.4),height ))
        d = np.sqrt(x*x+y*y)
        sigma, mu = width/sigma1, width/mu1
    #    sigma=0, mu=0
        g0 = (np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ))
        g0=np.fix(g0*input_thershold)
    #    print('np.max(g0) =',np.max(g0),'np.min(g0) =',np.min(g0))
        g1=np.uint8(g0)
    #    print('np.max(g1) =',np.max(g1),'np.min(g1) =',np.min(g1))
        g=np.max(g1) -g1  #cv.imshow('ss',ss)
        #cv.imshow('g0',g0),cv.imshow('g1',g1)
        #cv.imshow('g',g)
    #    print(np.max(g)) #cv.imshow('g1',g1)
#        th1=th/255
        th2=10*(th)/255
#        image21 = cv.addWeighted(g, 1, img[:,:,0], 1, 0.0)
#        image22 = cv.addWeighted(g, 1, img[:,:,1], 1, 0.0)
#        image23 = cv.addWeighted(g, 1, img[:,:,2], 1, 0.0)
        
        image21 = cv.addWeighted(g,   th2, img[:,:,0], 1, 0.0)
        image22 = cv.addWeighted(g,   th2, img[:,:,1], 1, 0.0)
        image23 = cv.addWeighted(g,   th2, img[:,:,2], 1, 0.0)
        
        #cv.imshow('image21',image21)
        #cv.imshow('image22',image22)
        #cv.imshow('image23',image23)
        image2=img;
        image2[:,:,0]=image21
        image2[:,:,1]=image22
        image2[:,:,2]=image23
#        cv.imshow('image2',image2)
        vis = np.concatenate((image2, img), axis=1)
#        cv.waitKey(300)
    print('finish shadow removal')
#    cv.waitKey(0)
    #cv.destroyAllWindows() 
    return ( image2 )   
# =============================================================================
# hair_removal
# =============================================================================

def hair_removal (I1):
    import cv2 as cv
    import numpy as np
    input_thershold=50
    sigma1=9
    mu1=11
    sigma1=7
    mu1=11
    img0 =I1# cv.imread('image/skin.jpg',1)
    img=img0  
    for i in range(25):
        input_thershold=i*2.5/10
    #    img = cv.imread('skin (1).bmp',1)
        height, width = img.shape[:2]
#        cv.imshow('img',img)  #a, b, c=img.size   #a =np.uint16(a)    #minn=0  #maxx=100
        x, y = np.meshgrid(np.linspace(-(width/2),(width/2),width), np.linspace(-(height/1.4),(height/1.4),height ))
        d = np.sqrt(x*x+y*y)
        sigma, mu = width/sigma1, width/mu1
    #    sigma=0, mu=0
        g0 = (np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ))
        g0=np.fix(g0*input_thershold)
        g1=np.uint8(g0)
        g=np.max(g1) -g1  #cv.imshow('ss',ss)
        image21 = cv.addWeighted(g, 1, img[:,:,0], 1, 0.0)
        image22 = cv.addWeighted(g, 1, img[:,:,1], 1, 0.0)
        image23 = cv.addWeighted(g, 1, img[:,:,2], 1, 0.0)
        image2=img;
        image2[:,:,0]=image21
        image2[:,:,1]=image22
        image2[:,:,2]=image23
#        cv.imshow('image2',image2)
        vis = np.concatenate((image2, img), axis=1)
    
        cv.waitKey(300)
    print('finish hair removal')
    cv.waitKey(0)    #cv.destroyAllWindows() 
    I2=image2
    return ( I2 )    


def hair_removal_exact(I2):
    
    import logging
    from hair_removal import remove_and_inpaint
    import sys 
    import time 
    import cv2 as cv
    import numpy as np
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Show logs
    dermoscopic_image_as_rgb = I2#cv.imread('image/skin.jpg',1)
    #cv.waitKey(0)
    t = time.time()
    # do stuff
#    cv.imshow('dermoscopic_image_as_rgb',dermoscopic_image_as_rgb)
    c = cv.waitKey(1000)
    #cv.waitKey(0)
    hairless_image, steps = remove_and_inpaint(dermoscopic_image_as_rgb)
#    cv.imshow('hairless_image',hairless_image)
    data = hairless_image / hairless_image.max() #normalizes data in range 0 - 255
    data = 255 * data
    hairless_image2 = data.astype(np.uint8)
    #cv.imshow("Window", hairless_image)
    cv.imwrite('hairless_image.jpg',hairless_image2)
    #cv.waitKey(0)
    elapsed = time.time() - t 
    print('time = ',elapsed)    
    c = cv.waitKey(5000)
#    cv.destroyAllWindows()
    I3 =hairless_image2
    return ( I3 ) 
# =========================================            ====================================
# edge_enhancement
# =============================================================================
def edge_enhancement(I2):
    import numpy as np
    import cv2;kernel = np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
    image2 = cv2.filter2D(I2, -1, kernel)
    return ( image2 ) 
    

