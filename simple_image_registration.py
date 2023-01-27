import numpy as np

def shift_image(image, colShift, rowShift, phaseDiff):
    bufft = np.fft.fft2(image)
    nr,nc = bufft.shape
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr/2.0),np.ceil(nr/2.0)));
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc/2.0),np.ceil(nc/2.0)));
    [Nc,Nr] = np.meshgrid(Nc,Nr)
    greg = bufft*np.exp(np.complex(0,1)*2*np.pi*(-rowShift*Nr/nr-colShift*Nc/nc));
    greg = greg*np.exp(np.complex(0,1)*phaseDiff);

    if np.can_cast(np.float32,image.dtype): # need to check this, too
        shiftedImage = np.abs(np.fft.ifft2(greg))
    else:
        shiftedImage = np.round(np.abs(np.fft.ifft2(greg))).astype(image.dtype)
    return shiftedImage

def register_images(refImage, shiftedImage, return_error=False):
    """
    shiftedBackImage, colShift, rowShift, phaseDiff, [error] = register_images(refImage, shiftedImage, return_error=False)
    
    Translated from MATLAB code written by Manuel Guizar
    downloaded from http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
    BUT I have not implemented sub-pixel registration. This code only registers to nearest pixel.
    """
    buf1ft = np.fft.fft2(refImage)
    buf2ft = np.fft.fft2(shiftedImage)
    m,n = buf1ft.shape
    CC = np.fft.ifft2(buf1ft*buf2ft.conjugate())

    max1 = np.max(CC,axis=0)
    loc1 = np.argmax(CC,axis=0)
    cloc = np.argmax(max1)

    rloc = loc1[cloc]
    CCmax = CC[rloc,cloc]
    
    if return_error:
        rfzero = np.sum(np.abs(buf1ft.ravel())**2)/(m*n)
        rgzero = np.sum(np.abs(buf2ft.ravel())**2)/(m*n)

        error = 1.0 - CCmax*CCmax.conjugate()/(rgzero*rfzero)
        error = np.sqrt(np.abs(error));

    #phaseDiff=np.arctan2(CCmax.imag,CCmax.real);
    phaseDiff=np.angle(CCmax);

    md2 = np.fix(m/2.0) #should this be float?
    nd2 = np.fix(n/2.0)
    if rloc > md2:
        rowShift = rloc - m; #CHECK!
    else:
        rowShift = rloc;#CHECK!

    if cloc > nd2:
        colShift = cloc - n;#CHECK!
    else:
        colShift = cloc;#CHECK!

    # Compute registered version of buf2ft
    nr,nc = buf2ft.shape
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr/2.0),np.ceil(nr/2.0)));
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc/2.0),np.ceil(nc/2.0)));
    [Nc,Nr] = np.meshgrid(Nc,Nr)
    greg = buf2ft*np.exp(np.complex(0,1)*2*np.pi*(-rowShift*Nr/nr-colShift*Nc/nc));
    greg = greg*np.exp(np.complex(0,1)*phaseDiff);

    if np.can_cast(np.float32,shiftedImage.dtype): # need to check this, too
        shiftedBackImage = np.abs(np.fft.ifft2(greg))
    else:
        shiftedBackImage = np.round(np.abs(np.fft.ifft2(greg))).astype(shiftedImage.dtype)
    
    if return_error:
        output = [shiftedBackImage, colShift, rowShift, phaseDiff, error]
    else:
        output = [shiftedBackImage, colShift, rowShift, phaseDiff]
    return output
