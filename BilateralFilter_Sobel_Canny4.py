import numpy as np
from numpy.linalg import eig, inv
import statistics
import math
import cv2
from scipy import fftpack
from numpy import zeros
import matplotlib.pyplot as plt

def ascol( arr ):
    '''
    If the dimensionality of 'arr' is 1, reshapes it to be a column matrix (N,1).
    '''
    if len( arr.shape ) == 1: arr = arr.reshape( ( arr.shape[0], 1 ) )
    return arr
def asrow( arr ):
    '''
    If the dimensionality of 'arr' is 1, reshapes it to be a row matrix (1,N).
    '''
    if len( arr.shape ) == 1: arr = arr.reshape( ( 1, arr.shape[0] ) )
    return arr

def fitellipse( x, opt = 'nonlinear', **kwargs ):
    '''
    function [z, a, b, alpha] = fitellipse(x, varargin)
    %FITELLIPSE   least squares fit of ellipse to 2D data
    %
    %   [Z, A, B, ALPHA] = FITELLIPSE(X)
    %       Fit an ellipse to the 2D points in the 2xN array X. The ellipse is
    %       returned in parametric form such that the equation of the ellipse
    %       parameterised by 0 <= theta < 2*pi is:
    %           X = Z + Q(ALPHA) * [A * cos(theta); B * sin(theta)]
    %       where Q(ALPHA) is the rotation matrix
    %           Q(ALPHA) = [cos(ALPHA), -sin(ALPHA); 
    %                       sin(ALPHA), cos(ALPHA)]
    %
    %       Fitting is performed by nonlinear least squares, optimising the
    %       squared sum of orthogonal distances from the points to the fitted
    %       ellipse. The initial guess is calculated by a linear least squares
    %       routine, by default using the Bookstein constraint (see below)
    %
    %   [...]            = FITELLIPSE(X, 'linear')
    %       Fit an ellipse using linear least squares. The conic to be fitted
    %       is of the form
    %           x'Ax + b'x + c = 0
    %       and the algebraic error is minimised by least squares with the
    %       Bookstein constraint (lambda_1^2 + lambda_2^2 = 1, where 
    %       lambda_i are the eigenvalues of A)
    %
    %   [...]            = FITELLIPSE(..., 'Property', 'value', ...)
    %       Specify property/value pairs to change problem parameters
    %          Property                  Values
    %          =================================
    %          'constraint'              {|'bookstein'|, 'trace'}
    %                                    For the linear fit, the following
    %                                    quadratic form is considered
    %                                    x'Ax + b'x + c = 0. Different
    %                                    constraints on the parameters yield
    %                                    different fits. Both 'bookstein' and
    %                                    'trace' are Euclidean-invariant
    %                                    constraints on the eigenvalues of A,
    %                                    meaning the fit will be invariant
    %                                    under Euclidean transformations
    %                                    'bookstein': lambda1^2 + lambda2^2 = 1
    %                                    'trace'    : lambda1 + lambda2     = 1
    %
    %           Nonlinear Fit Property   Values
    %           ===============================
    %           'maxits'                 positive integer, default 200
    %                                    Maximum number of iterations for the
    %                                    Gauss Newton step
    %
    %           'tol'                    positive real, default 1e-5
    %                                    Relative step size tolerance
    %   Example:
    %       % A set of points
    %       x = [1 2 5 7 9 6 3 8; 
    %            7 6 8 7 5 7 2 4];
    % 
    %       % Fit an ellipse using the Bookstein constraint
    %       [zb, ab, bb, alphab] = fitellipse(x, 'linear');
    %
    %       % Find the least squares geometric estimate       
    %       [zg, ag, bg, alphag] = fitellipse(x);
    %       
    %       % Plot the results
    %       plot(x(1,:), x(2,:), 'ro')
    %       hold on
    %       % plotellipse(zb, ab, bb, alphab, 'b--')
    %       % plotellipse(zg, ag, bg, alphag, 'k')
    % 
    %   See also PLOTELLIPSE
    
    % Copyright Richard Brown, this code can be freely used and modified so
    % long as this line is retained
    '''
    #error(nargchk(1, 5, nargin, 'struct'))
    
    x = asarray( x )
    
    ## Parse inputs
    # ...
    ## Default parameters
    kwargs[ 'fNonlinear' ] = opt is not 'linear'
    kwargs.setdefault( 'constraint', 'bookstein' )
    kwargs.setdefault( 'maxits', 200 )
    kwargs.setdefault( 'tol', 1e-5 )
    if x.shape[1] == 2:
        x = x.T
    if x.shape[1] < 6:
        raise RuntimeError('fitellipse:InsufficientPoints At least 6 points required to compute fit')
    
    ## Constraints are Euclidean-invariant, so improve conditioning by removing
    ## centroid
    centroid = mean(x, 1)
    x        = x - centroid.reshape((2,1))
    
    ## Obtain a linear estimate
    if kwargs['constraint'] == 'bookstein':
        ## Bookstein constraint : lambda_1^2 + lambda_2^2 = 1
        z, a, b, alpha = fitbookstein(x)
    
    elif kwargs['constraint'] == 'trace':
        ## 'trace' constraint, lambda1 + lambda2 = trace(A) = 1
        z, a, b, alpha = fitggk(x)
    
    ## Minimise geometric error using nonlinear least squares if required
    if kwargs['fNonlinear']:
        ## Initial conditions
        z0     = z
        a0     = a
        b0     = b
        alpha0 = alpha
        
        ## Apply the fit
        z, a, b, alpha, fConverged = fitnonlinear(x, z0, a0, b0, alpha0, **kwargs)
        
        ## Return linear estimate if GN doesn't converge
        if not fConverged:
            print('fitellipse:FailureToConverge', 'Gauss-Newton did not converge, returning linear estimate')
            z = z0
            a = a0
            b = b0
            alpha = alpha0
    
    ## Add the centroid back on
    z = z + centroid
    
    return z, a, b, alpha

def fitbookstein(x):
    '''
    function [z, a, b, alpha] = fitbookstein(x)
    %FITBOOKSTEIN   Linear ellipse fit using bookstein constraint
    %   lambda_1^2 + lambda_2^2 = 1, where lambda_i are the eigenvalues of A
    '''
    
    ## Convenience variables
    m  = x.shape[1]
    x1 = x[0, :].reshape((1,m)).T
    x2 = x[1, :].reshape((1,m)).T
    
    ## Define the coefficient matrix B, such that we solve the system
    ## B *[v; w] = 0, with the constraint norm(w) == 1
    B = hstack([ x1, x2, ones((m, 1)), power( x1, 2 ), multiply( sqrt(2) * x1, x2 ), power( x2, 2 ) ])
    
    ## To enforce the constraint, we need to take the QR decomposition
    Q, R = linalg.qr(B)
    
    ## Decompose R into blocks
    R11 = R[0:3, 0:3]
    R12 = R[0:3, 3:6]
    R22 = R[3:6, 3:6]
    
    ## Solve R22 * w = 0 subject to norm(w) == 1
    U, S, V = linalg.svd(R22)
    V = V.T
    w = V[:, 2]
    
    ## Solve for the remaining variables
    v = dot( linalg.solve( -R11, R12 ), w )
    
    ## Fill in the quadratic form
    A        = zeros((2,2))
    A.ravel()[0]     = w.ravel()[0]
    A.ravel()[1:3] = 1 / sqrt(2) * w.ravel()[1]
    A.ravel()[3]     = w.ravel()[2]
    bv       = v[0:2]
    c        = v[2]
    
    ## Find the parameters
    z, a, b, alpha = conic2parametric(A, bv, c)
    
    return z, a, b, alpha

def fitggk(x):
    '''
    function [z, a, b, alpha] = fitggk(x)
    % Linear least squares with the Euclidean-invariant constraint Trace(A) = 1
    '''
    
    ## Convenience variables
    m  = x.shape[1]
    x1 = x[0, :].reshape((1,m)).T
    x2 = x[1, :].reshape((1,m)).T
    
    ## Coefficient matrix
    B = hstack([ multiply( 2 * x1, x2 ), power( x2, 2 ) - power( x1, 2 ), x1, x2, ones((m, 1)) ])
    
    v = linalg.lstsq( B, -power( x1, 2 ), rcond=None,  )[0].ravel()
    
    ## For clarity, fill in the quadratic form variables
    A        = zeros((2,2))
    A[0,0]   = 1 - v[1]
    A.ravel()[1:3] = v[0]
    A[1,1]   = v[1]
    bv       = v[2:4]
    c        = v[4]
    
    ## find parameters
    z, a, b, alpha = conic2parametric(A, bv, c)
    
    return z, a, b, alpha


def fitnonlinear(x, z0, a0, b0, alpha0, **params):
    '''
    function [z, a, b, alpha, fConverged] = fitnonlinear(x, z0, a0, b0, alpha0, params)
    % Gauss-Newton least squares ellipse fit minimising geometric distance 
    '''
    
    ## Get initial rotation matrix
    Q0 = array( [[ cos(alpha0), -sin(alpha0) ], [ sin(alpha0), cos(alpha0) ]] )
    m = x.shape[1]
    
    ## Get initial phase estimates
    phi0 = angle( dot( dot( array([1, 1j]), Q0.T ), x - z0.reshape((2,1)) ) ).T
    u = hstack( [ phi0, alpha0, a0, b0, z0 ] ).T
    
    
    def sys(u):
        '''
        function [f, J] = sys(u)
        % SYS : Define the system of nonlinear equations and Jacobian. Nested
        % function accesses X (but changeth it not)
        % from the FITELLIPSE workspace
        '''
        
        ## Tolerance for whether it is a circle
        circTol = 1e-5
        
        ## Unpack parameters from u
        phi   = u[:-5]
        alpha = u[-5]
        a     = u[-4]
        b     = u[-3]
        z     = u[-2:]
        
        ## If it is a circle, the Jacobian will be singular, and the
        ## Gauss-Newton step won't work. 
        ##TODO: This can be fixed by switching to a Levenberg-Marquardt
        ##solver
        if abs(a - b) / (a + b) < circTol:
            print('fitellipse:CircleFound', 'Ellipse is near-circular - nonlinear fit may not succeed')
        
        ## Convenience trig variables
        c = cos(phi)
        s = sin(phi)
        ca = cos(alpha)
        sa = sin(alpha)
        
        ## Rotation matrices
        Q    = array( [[ca, -sa],[sa, ca]] )
        Qdot = array( [[-sa, -ca],[ca, -sa]] )
        
        ## Preallocate function and Jacobian variables
        f = zeros(2 * m)
        J = zeros((2 * m, m + 5))
        for i in range( m ):
            rows = range( (2*i), (2*i)+2 )
            ## Equation system - vector difference between point on ellipse
            ## and data point
            f[ rows ] = x[:, i] - z - dot( Q, array([ a * cos(phi[i]), b * sin(phi[i]) ]) )
            
            ## Jacobian
            J[ rows, i ] = dot( -Q, array([ -a * s[i], b * c[i] ]) )
            J[ rows, -5: ] = \
                hstack([ ascol( dot( -Qdot, array([ a * c[i], b * s[i] ]) ) ), ascol( dot( -Q, array([ c[i], 0 ]) ) ), ascol( dot( -Q, array([ 0, s[i] ]) ) ), array([[-1, 0],[0, -1]]) ])
        
        return f,J
    
    
    ## Iterate using Gauss Newton
    fConverged = False
    for nIts in range( params['maxits'] ):
        ## Find the function and Jacobian
        f, J = sys(u)
        
        ## Solve for the step and update u
        #h = linalg.solve( -J, f )
        h = linalg.lstsq( -J, f, rcond=None)[0]
        u = u + h
        
        ## Check for convergence
        delta = linalg.norm(h, inf) / linalg.norm(u, inf)
        if delta < params['tol']:
            fConverged = True
            break
    
    alpha = u[-5]
    a     = u[-4]
    b     = u[-3]
    z     = u[-2:]
    
    return z, a, b, alpha, fConverged

def conic2parametric(A, bv, c):
    '''
    function [z, a, b, alpha] = conic2parametric(A, bv, c)
    '''
    ## Diagonalise A - find Q, D such at A = Q' * D * Q
    D, Q = linalg.eig(A)
    Q = Q.T
    
    ## If the determinant < 0, it's not an ellipse
    if prod(D) <= 0:
        raise RuntimeError('fitellipse:NotEllipse Linear fit did not produce an ellipse')
    
    ## We have b_h' = 2 * t' * A + b'
    t = -0.5 * linalg.solve(A, bv)
    
    c_h = dot( dot( t.T, A ), t ) + dot( bv.T, t ) + c
    
    z = t
    a = sqrt(-c_h / D[0])
    b = sqrt(-c_h / D[1])
    alpha = atan2(Q[0,1], Q[0,0])
    
    return z, a, b, alpha


def tlFunction():
    global nextPixel
    global curh
    global curw
    global primaryDir
    global TL
    nextPixel = True
    curh-=1
    curw-=1
    primaryDir = TL

def tmFunction():
    global nextPixel
    global curh
    global primaryDir
    global TM
    nextPixel = True
    curh-=1
    primaryDir = TM

def trFunction():
    global nextPixel
    global curh
    global curw
    global primaryDir
    global TR
    nextPixel = True
    curh-=1
    curw+=1
    primaryDir = TR

def mrFunction():
    global nextPixel
    global curw
    global primaryDir
    global MR
    nextPixel = True
    curw+=1
    primaryDir = MR

def brFunction():
    global nextPixel
    global curh
    global curw
    global primaryDir
    global BR
    nextPixel = True
    curh+=1
    curw+=1
    primaryDir = BR

def bmFunction():
    global nextPixel
    global curh
    global primaryDir
    global BM
    nextPixel = True
    curh+=1
    primaryDir = BM

def blFunction():
    global nextPixel
    global curh
    global curw
    global primaryDir
    global BL
    nextPixel = True
    curh+=1
    curw-=1
    primaryDir = BL

def mlFunction():
    global nextPixel
    global curw
    global primaryDir
    global BR
    nextPixel = True
    curw-=1
    primaryDir = ML

#####################################################################
# The original image
#####################################################################

# read image
img = plt.imread('resize.bmp')
#img =cv2.resize(img,(800,800))
#bilateral_filtered_image = cv2.bilateralFilter(img, 5, 175, 175)

img = img/255
thold = 15
plt.figure()
plt.title('Raw')
plt.imshow(img)

#bilateral_filtered_image = bilateral_filtered_image/255
#plt.figure()
#plt.title('Bilateral CV2')
#plt.imshow(bilateral_filtered_image)

width, height, channels = img.shape

imga = zeros([width,height,3])
cannyImage = zeros([width,height])
contourImage = zeros([width,height])
imgDir = zeros([width,height])
imgSobl = imga

t = imga[1,1]
sobelX =[[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]
sobelY =[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]
cannyArray = zeros([3,3])
imgSob = 0.0
tS =0.0
tSx = tS
tSy = tS

tr = t

mat = 5
mp =1+(int)(mat/2)
ta = zeros([mat,mat,3])
#plt.figure()
#plt.imshow(imga)
#####################################################################
# Prepare an Gaussian convolution kernel
#####################################################################

# First a 1-D  Gaussian
t = np.linspace(-mat, mat, mat)  # esecentialy for loop starting at -10 ending at 10 with 30 steps

bump = np.exp(-0.1*t**2)  # Used to generate line that erodes in an expodential patern -10 to + 10 building a bell curve
bump /= np.trapz(bump) # normalize the integral to 1

#print ('bump',bump)

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

#print ('kernel',kernel)
for xk in range (0,(mat-1), 1):  # nested loop to parce through raw image	
    for yk in range (0,(mat-1), 1):
        ta[xk,yk,0] = kernel[xk,yk]
        ta[xk,yk,1] = kernel[xk,yk]
        ta[xk,yk,2] = kernel[xk,yk]

for h in range (0,(height-1), 1):  #Initial threshold and greyscale conversion
    for w in range (0,(width-1), 1):
        img[h,w] = (img[h,w,0] + img[h,w,1] + img[h,w,2])/3
        #		print(img[h,w])
        if img[h,w,0] < 0.02:
            img[h,w] = [0,0,0]

for h in range (mp,(height-mp), 1):  # Smothing filter	
    for w in range (mp,(width-mp), 1):
        tr = img[h,w]
        #		trAvg = (tr[0]+tr[1]+tr[2])/3
        if tr[0] > 0.0:
            for xk in range (0,(mat-1), 1):  # nested loop to parce through raw image
                for yk in range (0,(mat-1), 1):
                    intPixel = ((img[(h+(xk-mp)),(w+(yk-mp)),0])+(img[(h+(xk-mp)),(w+(yk-mp)),1])+(img[(h+(xk-mp)),(w+(yk-mp)),2]))/3
                    #					if (intPixel -(thold/255)) <= trAvg <= (intPixel + (thold/255)):
                    t[0] += img[(h+(xk-mp)),(w+(yk-mp)),0]*ta[xk,yk,0]
                    t[1] += img[(h+(xk-mp)),(w+(yk-mp)),1]*ta[xk,yk,1]
                    t[2] += img[(h+(xk-mp)),(w+(yk-mp)),2]*ta[xk,yk,2]
        #					else:
        #						t[0] += tr[0]*ta[xk,yk,0]
        #						t[1] += tr[1]*ta[xk,yk,1]
        #						t[2] += tr[2]*ta[xk,yk,2]
        else:
            t[0] = 0
            t[1] = 0
            t[2] = 0

        #		print(intPixel)
        imga[h,w,0] = t[0]
        imga[h,w,1] = t[1]
        imga[h,w,2] = t[2]
        t[0] = 0
        t[1] = 0
        t[2] = 0

plt.figure()
plt.imshow(imga)
plt.title('Bilateral filter')

imga1 = np.uint8(imga*255)
cv2.imshow('Filtered Image ', imga1)
#cv2.waitKey(0)
for h in range (1,(height-2), 1):  # Sobel filter
    for w in range (1,(width-2), 1):
        if imga[h,w,0] > 0.0:
            #			tS = imga1[h,w,0]
            for xk in range (0,3,1):  # nested loop to parce through raw image
                for yk in range (0,3,1):
                    tSx+=((sobelX[xk][yk])*imga1[((h-1)+xk),((w-1)+yk),0])
                    tSy+=((sobelY[xk][yk])*imga1[((h-1)+xk),((w-1)+yk),0])
            tS =math.sqrt(tSx*tSx + tSy*tSy)
            #			print('tS :',tS)
            #			tSx = abs(tSx)
            #			tSy = abs(tSy)
            if tS >76:
                imgSobl[h,w] =[tS,tS,tS]
                #				if tSx > 0.0:
                dirE = ((math.atan(tSy/tSx))*180/(math.pi))
                #					print('Dir:',dirE)
                if dirE < 0:
                    dirE = dirE + 180
                #				print('Dir:',dirE)
                dirE = 180-dirE
                imgDir[h,w] = dirE
            else:
                imgSobl[h,w] = [0,0,0]
        tS = 0.0
        tSx = 0.0
        tSy = 0.0

cv2.imwrite('sobel.jpg',imgSobl)
cv2.imwrite('dirE.jpg',imgDir)

for h in range (1,(height-2), 1):  # Canny filter,  modifed so that direction is coded into solution only requires 3 bits per pixel,  will speed up contour identification
    for w in range (1,(width-2), 1):
        dirE = imgDir[h,w]
        maxPoint = True
        maxDir = 0
        curPoint = imgSobl[h,w,0]
        if curPoint > 0.0:
            if 22.5 < dirE <= 67.5:#NW
                maxDir = 1
                #				print('122-157',dirE)
                if (imgSobl[h-1,w+1,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h+1,w-1,0]):
                    maxPoint = False
            elif 67.5 < dirE <=122.5:#N
                #				print('67-112 :',dirE,)
                maxDir = 1
                if (imgSobl[h-1,w,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h+1,w,0]):
                    maxPoint = False
            elif 122.5 < dirE <= 157.5:#NE
                maxDir = 1
                #				print('22-67',dirE)
                if (imgSobl[h-1,w-1,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h+1,w+1,0]):
                    maxPoint = False
            else:#E
                #				print('All else',dirE)
                maxDir = 1
                if (imgSobl[h,w-1,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h,w+1,0]):
                    maxPoint = False

            if maxPoint:
                cannyImage[h,w] = maxDir
        #	print('Current point',dirE)

contourList = [[0 for i in range(3)] for j in range(5000)]
contNum = 128
contPixCount = 0
curContNum = 0
contPlace = 0
oldContPlace = 0
ND=0  #directions for 8 connected matrix
TL=1
TM=2
TR=3
MR=4
BR=5
BM=6
BL=7
ML=8

nextPixel = False
contDirection = 0#  0 for CW,  1 for CWW 2 for end of contour
for h in range (1,(height-1), 1): #parces through modified canny 
    for w in range (1,(width-1), 1):
        cannyImgPix = cannyImage[h,w]
        curh = h
        curw = w
        primaryDir = ND

        while ((curh <= (height-1)) and (curw< (width - 1)) and (cannyImgPix == 1)):
            nextPixel = False
            curhtemp = curh
            curwtemp = curw
            tl = cannyImage[curh - 1, curw - 1] #defining the 8 connected matrix
            tm = cannyImage[curh - 1, curw]	#top left TL, top Middle ect....
            tr = cannyImage[curh - 1, curw + 1]
            ml = cannyImage[curh, curw - 1]
            mr = cannyImage[curh, curw + 1]
            bl = cannyImage[curh + 1, curw - 1]
            bm = cannyImage[curh + 1, curw]
            br = cannyImage[curh + 1, curw + 1]
            if tl == 1 or tm == 1 or tr == 1 or ml == 1 or mr == 1 or bl == 1 or bm == 1 or br == 1:

#       ----------------- MR First ---------------------
                if primaryDir == ND or primaryDir == MR:
                    if (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    elif (br == 1):
                        brFunction()
                    elif (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (ml == 1):
                        mlFunction()
                    else:
                        nextPixel = False
#       ----------------- BR ---------------------
                elif primaryDir == BR:
                    if (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    elif (br == 1):
                        brFunction()
                    elif (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (ml == 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    else:
                        nextPixel = False

#       ----------------- BM ---------------------
                elif primaryDir == BM:
                    if (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    elif (br == 1):
                        brFunction()
                    elif (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (ml == 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    else:
                        nextPixel = False

#       ----------------- BL ---------------------
                elif primaryDir == BL:
                    if (mr == 1):
                        mrFunction()
                    elif (br == 1):
                        brFunction()
                    elif (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (ml == 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    else:
                        nextPixel = False

#----------------- ML ---------------------
                elif primaryDir == ML:
                    if (bl == 1):
                        blFunction()
                    elif (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (ml== 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    else:
                        nextPixel = False

#       ----------------- TL ---------------------
                elif primaryDir == TL:
                    if (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (ml == 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    elif (br == 1):
                        brFunction()
                    else:
                        nextPixel = False


#       ----------------- TM ---------------------
                elif primaryDir == TM:
                    if (bl == 1):
                        blFunction()
                    elif (ml == 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    elif (bl == 1):
                        blFunction()
                    elif (bm == 1):
                        bmFunction()
                    else:
                        nextPixel = False

# ----------------- TR ---------------------
                elif primaryDir == TR:
                    if (ml == 1):
                        mlFunction()
                    elif (tl == 1):
                        tlFunction()
                    elif (tm == 1):
                        tmFunction()
                    elif (tr == 1):
                        trFunction()
                    elif (mr == 1):
                        mrFunction()
                    elif (br == 1):
                        brFunction()
                    elif (bm == 1):
                        bmFunction()
                    elif (bl == 1):
                        blFunction()
                    else:
                        nextPixel = False
                else:
                    cannyImage[curhtemp,curwtemp] = contNum
#                    cannyImage[curh, curw] = contNum
                    nextPixel = False

#            print('Position H:',curh, ' W:',curw, ' contNum:',contNum)
            if nextPixel == True:
                cannyImage[curhtemp,curwtemp] = contNum
                cannyImgPix = cannyImage[curh,curw]
                contourList[contPlace] = [contNum,curh,curw]
                contPlace+=1
                contPixCount+=1
            else:
                cannyImgPix = 0
                #cannyImage[curh,curw] =contNum
                cannyImage[curhtemp, curwtemp] = contNum
                contNum+=1
#                print('ContPixCount', contPixCount)
                if contPixCount < 80:
                    contPlace = oldContPlace
 #                   print('ContPlace',contPlace)
                contPixCount = 0
                oldContPlace = contPlace

#                contourList[contPlace] = {contNum,conh,conw}
#                    print('contourList ',contourList[contPlace])
#print(contourList)

#bilateral_filtered_image8 = np.uint8(bilateral_filtered_image*255)
#edge_detected_image1 = cv2.Canny((bilateral_filtered_image8), 75, 200)

#edge_detected_image1 = edge_detected_image1/255
#imgFP = img
#imgSobl = imgSobl/(imgSobl.max()/255.0)
cv2.imshow('Edge Sobel',imgSobl/255)
#cv2.waitKey(0)
#imgSobl = imgSobl/255
np.savetxt("contourList.csv", contourList, delimiter=",")
print(contourList)
value = 0
for i in contourList:
    h = i[1]
    w = i[2]
    value = i[0]
    contourImage[h, w] = value
#xList = []
#yList = []
#oldContourNumber = contourList[0][0]
#ellipseList = []
#nextContour = True
#for i in contourList:
#    if nextContour == False:
#        oldContourNumber = i[0]
#        nextContour = True
#    if oldContourNumber == i[0]:
#       xList.append([i[1],i[2]])
#        yList.append(i[2])
#    else:
#        print('Xlist ',xList)
#        print('YList ', yList)
#        x = np.array(xList)
#        y = np.array(yList)
#        print('Xlist ',x)
#        print('YList ', y)
#        a = fitEllipse(x,y)
#        center = ellipse_center(a)
#phi = ellipse_angle_of_rotation(a)
#        phi = ellipse_angle_of_rotation2(a)
#        axes = ellipse_axis_length(a)
#        print('Axes:',axes,' Angle:', phi, ' Centre:', center)  
#        ellipseProperties = [center,phi,axes]
#        ellipseList.append(ellipseProperties)
#        cv2.ellipse(img,(centre),(100,50),0,0,360,255,-1)
#        nextContour = False
#        xList = []
#        yList = []

#print(ellipseList)

plt.figure()
plt.imshow(contourImage)
plt.title('contourImage')
cv2.imwrite('contourImage.jpg',contourImage)
plt.figure()
plt.imshow(cannyImage/255)
cv2.imwrite('cannyImage.jpg',cannyImage);
plt.title('Canny Filter')
plt.figure()
plt.imshow(imgDir/255)
plt.title('ImageDir')
print('D-type: ',cannyImage.dtype)
print('Shape: ',cannyImage.shape)
cannyImage = cannyImage.astype(np.uint8)
#plt.figure()
#plt.imshow(imgDir/255)
#plt.title('direction')
#imgConv= cv2.cvtColor(cannyImage, cv2.COLOR_RGB2GRAY)

plt.show()

