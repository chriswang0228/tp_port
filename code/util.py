import cv2
import numpy as np
from skimage import io, measure, morphology
from math import radians, sin, cos, tan, log, dist
from sklearn.neighbors import KDTree
import torch
from torchvision import transforms as T
from plantcv import plantcv as pcv
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lonlat_to_97(lon,lat):
    """
    It transforms longitude, latitude to TWD97 system.

    Parameters
    ----------
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees 

    Returns
    -------
    x, y [TWD97]
    """
    
    lat = radians(lat)
    lon = radians(lon)
    
    a = 6378137.0
    b = 6356752.314245
    long0 = radians(121)
    k0 = 0.9999
    dx = 250000

    e = (1-b**2/a**2)**0.5
    e2 = e**2/(1-e**2)
    n = (a-b)/(a+b)
    nu = a/(1-(e**2)*(sin(lat)**2))**0.5
    p = lon-long0

    A = a*(1 - n + (5/4.0)*(n**2 - n**3) + (81/64.0)*(n**4  - n**5))
    B = (3*a*n/2.0)*(1 - n + (7/8.0)*(n**2 - n**3) + (55/64.0)*(n**4 - n**5))
    C = (15*a*(n**2)/16.0)*(1 - n + (3/4.0)*(n**2 - n**3))
    D = (35*a*(n**3)/48.0)*(1 - n + (11/16.0)*(n**2 - n**3))
    E = (315*a*(n**4)/51.0)*(1 - n)

    S = A*lat - B*sin(2*lat) + C*sin(4*lat) - D*sin(6*lat) + E*sin(8*lat)

    K1 = S*k0
    K2 = k0*nu*sin(2*lat)/4.0
    K3 = (k0*nu*sin(lat)*(cos(lat)**3)/24.0) * (5 - tan(lat)**2 + 9*e2*(cos(lat)**2) + 4*(e2**2)*(cos(lat)**4))

    y_97 = K1 + K2*(p**2) + K3*(p**4)

    K4 = k0*nu*cos(lat)
    K5 = (k0*nu*(cos(lat)**3)/6.0) * (1 - tan(lat)**2 + e2*(cos(lat)**2))

    x_97 = K4*p + K5*(p**3) + dx
    return x_97, y_97

def mapping(src, dst):
    dst_img = cv2.cvtColor(dst.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    src_img = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # Initiate ORB detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with ORB
    kp_src, des_src = sift.detectAndCompute(src_img,None)
    kp_dst, des_dst = sift.detectAndCompute(dst_img,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_src,des_dst,k=2)
    good = []
    for m in matches:
        if (m[0].distance < 0.8*m[1].distance):
            good.append(m)
    matches = np.asarray(good)
    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp_src[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp_dst[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        src_kp = np.array([
            [src[0][0][0],src[0][0][1], 1],
            [src[3][0][0],src[3][0][1], 1]
        ]).T
        dst_kp = H.dot(src_kp)
        dst_kp = (dst_kp/dst_kp[2]).T[:, :2]
        dst_kp[:,[1,0]] = dst_kp[:,[0,1]]
        scale_rate = dist(dst_kp[0], dst_kp[1])/dist(src[0][0], src[3][0])
        
    return H, scale_rate

def img2otho(img, exif, otho_img):
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    point_pair=np.array([[[5442, 0], [2784295.23560,289479.57382]],
    [[574, 6298], [2784417.268382,289637.447092]],
    [[30393, 34491], [2783669.75993,290344.20025]],
    [[35835, 28735], [2783533.338170,290199.908113]]])
    point_pair = point_pair.transpose([1,0,2])
    img_pts = np.float32(point_pair[0])
    twd_pts = np.float32(point_pair[1])
    H_twd2otho, _ = cv2.findHomography(twd_pts.reshape([4,1,2]), img_pts.reshape([4,1,2]), cv2.RANSAC, 5.0)
    exif_data = exif
    lat = exif_data[34853][2]
    lon = exif_data[34853][4]
    lat = float(lat[0])+float(lat[1])/60+float(lat[2])/3600
    lon = float(lon[0])+float(lon[1])/60+float(lon[2])/3600
    lat, lon = lonlat_to_97(lon,lat)
    gps_point = np.array([lon,lat,1])
    gps_point = H_twd2otho.dot(gps_point)
    gps_point = (gps_point/gps_point[2]).T[:2].astype(np.int32)
    dst_img = otho_img[gps_point[0]-500:gps_point[0]+500, gps_point[1]-500:gps_point[1]+500]
    H_img2otho, scale_rate = mapping(np_img, dst_img)
    dst = cv2.warpPerspective(np_img, H_img2otho, (dst_img.shape[1],dst_img.shape[0]))
    
    return dst, H_img2otho, scale_rate, gps_point[0], gps_point[1]

def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c # shift the points
    A = A.T #3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True) # A=u*s*vh
    normal = u[:,-1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal,normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u,s,c,normal


def calcu_dis_from_ctrlpts(ctrlpts):
    if ctrlpts.shape[1]==4:
        return np.sqrt(np.sum((ctrlpts[:,0:2]-ctrlpts[:,2:4])**2,axis=1))
    else:
        return np.sqrt(np.sum((ctrlpts[:,[0,2]]-ctrlpts[:,[3,5]])**2,axis=1))


def estimate_normal_for_pos(pos,points,n):
    # estimate normal vectors at a given point
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pos, k=1, return_distance=False, dualtree=False, breadth_first=False)
    if idx[0][0]<5:
      idx[0][0] = 5
    elif idx[0][0]>len(points)-6:
      idx[0][0] = len(points)-6
    idx = np.array([list(range(idx[0][0]-5,idx[0][0]+6))])
    #pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0,pos.shape[0]):
        pts_for_normals = pts[idx[i,:],:]
        _,_,_,normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals

def get_crack_ctrlpts(centers,normals,bpoints,hband=5,vband=2):
    # main algorithm to obtain crack width
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)

    xmatrix = np.array([[0,1],[-1,0]])
    cnormalsx = np.dot(xmatrix,cnormals.T).T # the normal of x axis
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx,ny])
            bpoints_loc = np.dot(tform,bpoints.T).T
            cpoints_loc = np.dot(tform,cpoints.T).T
            ci = cpoints_loc[i]

            bl_ind = (bpoints_loc[:,0]-(ci[0]-hband))*(bpoints_loc[:,0]-ci[0])<0
            br_ind = (bpoints_loc[:,0]-ci[0])*(bpoints_loc[:,0]-(ci[0]+hband))<=0
            bl = bpoints_loc[bl_ind] # left points
            br = bpoints_loc[br_ind] # right points

            blt = bl[bl[:,1]>ci[1]]
            blt = blt[blt[:,1]-ci[1]<10]

            blb = bl[bl[:,1]<ci[1]]
            blb = blb[blb[:,1]-ci[1]>-10]

            brt = br[br[:,1]>ci[1]]
            brt = brt[brt[:,1]-ci[1]<10]

            brb = br[br[:,1]<ci[1]]
            brb = brb[brb[:,1]-ci[1]>-10]


            #bh = np.vstack((bl,br))
            #bmax = np.max(bh[:,1])
            #bmin = np.min(bh[:,1])

            #blt = bl[bl[:,1]>bmax-vband] # left top points
            #blb = bl[bl[:,1]<bmin+vband] # left bottom points

            #brt = br[br[:,1]>bmax-vband] # right top points
            #brb = br[br[:,1]<bmin+vband] # right bottom points


            t1 = blt[np.argsort(blt[:,0])[-1]]
            t2 = brt[np.argsort(brt[:,0])[0]]

            b1 = blb[np.argsort(blb[:,0])[-1]]
            b2 = brb[np.argsort(brb[:,0])[0]]


            interp1 = (ci[0]-t1[0])*((t2[1]-t1[1])/(t2[0]-t1[0]))+t1[1]
            interp2 = (ci[0]-b1[0])*((b2[1]-b1[1])/(b2[0]-b1[0]))+b1[1]

            if interp1-ci[1]>0 and interp2-ci[1]<0:
                widths.append([i,interp1-ci[1],interp2-ci[1]])

                interps = np.array([[ci[0],interp1],[ci[0],interp2]])

                interps_rec = np.dot(np.linalg.inv(tform),interps.T).T

                #show_2dpoints([bpointsxl_loc1,bpointsxl_loc2,bpointsxr_loc1,bpointsxr_loc2,np.array([ptsl_1,ptsl_2]),np.array([ptsr_1,ptsr_2]),interps,ci.reshape(1,-1)],s=[1,1,1,1,20,20,20,20])
                interps_rec = interps_rec.reshape(1,-1)[0,:]
                interp_segm.append(interps_rec)
        except:
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    # check
    # show_2dpoints([np.array([[ci[0],interp1],[ci[0],interp2]]),np.array([t1,t2,b1,b2]),cpoints_loc,bl,br],[10,20,15,2,2])
    return interp_segm, widths

def estimate_pci(img, scale, road_area, dilation):    #1cm = 10pix, 則scale = 0.1
    widlist=[]
    estimate_step = int(20/scale)  #每隔20cm需量一次寬度，20cm對應的像素個數
    for i in range(1):
        arr=np.array(img)==1
        c=morphology.remove_small_objects(arr,min_size=100,connectivity=1)
        image=np.zeros(c.shape)
        image[c]=1
        iw,ih = image.shape
        blobs  = np.copy(image)
        blobs = blobs.astype(np.uint8)

        skeleton = pcv.morphology.skeletonize(mask=blobs)
        segmented_img, obj = pcv.morphology.segment_skeleton(skeleton)
        B = [1, 0]
        contours = measure.find_contours(blobs, 0.8)
        bpoints=[]
        for i in contours:
            for j in i:
                bpoints.append(j)
        bpoints=np.array(bpoints)

        widlist=[]

        for cent in obj:
            for i,cpoint in enumerate(cent):
                cent[i] = [j for _,j in sorted(zip(B,cpoint[0]))]
                
            centers=np.array(cent).reshape((len(cent),2))
            bpixel = np.zeros((iw,ih,3),dtype=np.uint8)
            bpoints = bpoints.astype(int)
            bpixel[bpoints[:,0],bpoints[:,1],0] = 255
            for i in range(int(estimate_step/2),len(centers)-int(estimate_step/2),estimate_step):
                pos = np.array(centers[i]).reshape(1,-1) # input (x,y) where need to calculate crack width

                posn = estimate_normal_for_pos(pos,centers,3)

                interps, widths2 = get_crack_ctrlpts(pos,posn,bpoints,hband=1.5,vband=2)
                if not widths2.all():
                    widlist.append(abs(widths2[0][1])+abs(widths2[0][2]))
        widlist=np.array(widlist)
    widlist=widlist*scale  #像素寬度轉換為公分
    widlist=widlist/dilation  #折減寬度以抵銷預測誤差
    h,m,l=0,0,0
    for wid in widlist:
        if wid < 0.3: #寬度小於0.3cm
            l=l+1
        elif wid>0.5: #寬度大於0.5cm
            h=h+1
        else:
            m=m+1
    [h,m,l] = np.array([h,m,l])*0.2 #因每個寬度都代表一個0.2m的裂縫，所以將各級裂縫的數量乘0.2m即可獲得總長。
    [h,m,l] = np.array([h,m,l])/road_area*100 #總長/面積取密度(%)。
    if l != 0:
        l=log(l,10)
        l=1.7+4.45*l+5.18*l**2

    if m != 0:
        m=log(m,10)
        m=2.1+11.51*m+4.93*m**2

    if h != 0:
        h=log(h,10)
        h=8.3+14.06*h+12.96*h**2

    todo_list=[]
    if h>2:
        todo_list.append(h)
    if m>2:
        todo_list.append(m)
    if l>2:
        todo_list.append(l)
    todo_list=np.array(sorted(todo_list))
    if len(todo_list)==3:
        total=np.sum(todo_list)
        cdv1=-6.4+0.82*total-0.0013*total**2
        todo_list[-1]=2
        total=np.sum(todo_list)
        cdv2=-3.6+0.91*total-0.0017*total**2
        todo_list[-2]=2
        total=np.sum(todo_list)
        cdv=np.max(np.array([cdv1,cdv2,total]))
    elif len(todo_list)==2:
        total=np.sum(todo_list)
        cdv1=-3.6+0.91*total-0.0017*total**2
        todo_list[-1]=2
        total=np.sum(todo_list)
        cdv=np.max(np.array([cdv1,total]))
    elif len(todo_list)==1:
        total=np.sum(todo_list)
        cdv=total
    else:
        cdv=0
    return 100-cdv

def get_road_mask(mask, clip_mask, H_img2otho, gps_point_x, gps_point_y):
    bound = np.array([
    [0, 0, 1],
    [mask.shape[1], 0, 1],
    [mask.shape[1], mask.shape[0], 1],
    [0, mask.shape[0], 1],
    
    ]).T
    mask_bound = H_img2otho.dot(bound)
    mask_bound = (mask_bound/mask_bound[2]).T[:, :2]
    mask_bound = mask_bound.astype(np.int32)+np.array([1500, 1500])
    mask_bound = (mask_bound/10).astype(np.int32)
    point_pair=np.array([[mask_bound[0], [0, 0]],
    [mask_bound[1], [mask.shape[1], 0]],
    [mask_bound[2], [mask.shape[1], mask.shape[0]]],
    [mask_bound[3], [0, mask.shape[0]]]])
    point_pair = point_pair.transpose([1,0,2])
    img_pts = np.float32(point_pair[0])
    twd_pts = np.float32(point_pair[1])
    H_o2i, _ = cv2.findHomography(img_pts.reshape([4,1,2]), twd_pts.reshape([4,1,2]), cv2.RANSAC, 5.0)
    road_mask = clip_mask[int(gps_point_x/10-200):int(gps_point_x/10+200), int(gps_point_y/10-200):int(gps_point_y/10+200)]
    road_mask = cv2.warpPerspective(road_mask, H_o2i, (int(mask.shape[1]),int(mask.shape[0])), flags=0)
    
    return road_mask

def correct_H(H, w, h):
    """ Correct the homography matrix so that the projected image is fully displayed

    Args:
        H (_type_): homography matrix between two images, (ndarray, (3, 3))
        w (_type_): image width
        h (_type_): image height

    Returns:
        H (_type_): correct homography matrix, (ndarray, (3, 3))
        correct_w: projection width
        correct_h: projection height
    """
    new_H = H.copy()
    corner_pts = np.array([[[0, 0], [w, 0], [0, h], [w, h]]], dtype=np.float32)
    min_out_w, min_out_h = cv2.perspectiveTransform(corner_pts, new_H)[0].min(axis=0).astype(np.int32)
    new_H[0, :] -= new_H[2, :] * min_out_w
    new_H[1, :] -= new_H[2, :] * min_out_h
    correct_w, correct_h = cv2.perspectiveTransform(corner_pts, new_H)[0].max(axis=0).astype(np.int32)

    return new_H, correct_w, correct_h

def pred_alg(img, model):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    images = []
    results=[]
    f=0
    for i in range(int(img.shape[0]/480)):
        for j in range(int(img.shape[1]/480)):
            images.append(img[480*i:480*i+480, 480*j:480*j+480])
            f+=1
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    for image in images:
        image = t(image)
        image=image.to(device)
        with torch.no_grad():
            image = image.unsqueeze(0)        
            output = model(image)
            result=torch.argmax(output, dim=1).squeeze(0).squeeze(0).cpu().numpy()
            results.append(result)
    mask = np.zeros(img.shape[:2])
    f=0
    for i in range(int(img.shape[0]/480)):
        for j in range(int(img.shape[1]/480)):
            mask[480*i:480*i+480, 480*j:480*j+480]=results[f]
            f+=1
    return mask

def pred_expan(H, img):
    theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi - 45
    if theta>90:
        theta-=90
    angle1 = theta
    angle2 = theta-90
    kernel_size = 1
    blur_gray = cv2.GaussianBlur(img.astype(np.uint8)*100,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 200  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1000  # minimum number of pixels making up a line
    max_line_gap = 500  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    w_problem = False
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            if abs(math.degrees(math.atan((x2-x1)/(y2-y1)))-angle1)>5 and abs(math.degrees(math.atan((x2-x1)/(y2-y1)))-angle2)>5:
                w_problem = True
    return w_problem, line_image