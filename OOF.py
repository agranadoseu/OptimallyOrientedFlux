"""
These functions implement an Optimally Oriented Flux (OOF) filter
This Python implementation is based on:
    - https://uk.mathworks.com/matlabcentral/fileexchange/41612-optimally-oriented-flux-oof-for-3d-curvilinear-structure-detection
    - https://github.com/fepegar/optimally-oriented-flux


Written by:
Alejandro Granados
School of Biomedical Engineering and Patient Sciences
King's College London, UK

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import math
import numpy as np
from scipy.special import jv    # Besssel functions 1st order

import SimpleITK as sitk



def oof3response(image=None, radii=[], resp_type=3):
    print('   ¬ Compute OOF filter response ...')

    # response_type = 3 :: sqrt(max(0, l1) .*max(0, l2));
    # OOF tensor eigenvalues :: l1 >> l2 >> l3
    # normalisation_type: blob-like (0), curvilinear (1), planar (2)
    # sigma: sigma >= min(radii), otherwise normalisation_type = 0
    opts = {'ntype': 1, 'sigma': min(image.GetSpacing()), 'use_absolute': True,
            'radii': radii, 'resp_type': resp_type}

    if min(radii)<opts['sigma'] and opts['ntype']>0:
        print('Normalisation type is set to zero since sigma<min(radii)')
        opts['ntype'] = 0

    # image
    data = sitk.GetArrayFromImage(image)
    size = [image.GetSize()[i] for i in [2,1,0]]
    spacing = [image.GetSpacing()[i] for i in [2,1,0]]

    # output
    output_data = np.zeros_like(data).astype('float64')

    # Fast Fourier Transform
    fft = np.fft.fftn(data)

    # Radius from Fourier coordinates
    x, y, z = ifft_shifted_coord_matrix(size, spacing)
    x /= size[0] * spacing[0]
    y /= size[1] * spacing[1]
    z /= size[2] * spacing[2]
    radius = np.sqrt(x**2 + y**2 + z**2) + 1e-12

    # iterate radii
    for r in radii:
        normalisation = 4/3 * math.pi * r**3 / \
                        (jv(1.5, 2*math.pi*r*1e-12) / 1e-18) / \
                        r**2 * \
                        (r / math.sqrt(2*r*opts['sigma'] - opts['sigma']**2))**opts['ntype']

        besseljBuffer = normalisation * \
                        np.exp(-opts['sigma']**2 * 2 * math.pi**2 * radius**2) / \
                        radius**(3/2)

        besseljBuffer = (np.sin(2*math.pi*r*radius) / (2*math.pi*r*radius) - np.cos(2*math.pi*r*radius)) * \
                        besseljBuffer * \
                        np.sqrt(1 / math.pi / math.pi / r / radius)

        besseljBuffer = besseljBuffer * fft

        # 3x3 symmetric matrix
        a11 = np.real(np.fft.ifftn(x * x * besseljBuffer))
        a12 = np.real(np.fft.ifftn(x * y * besseljBuffer))
        a13 = np.real(np.fft.ifftn(x * z * besseljBuffer))
        a22 = np.real(np.fft.ifftn(y * y * besseljBuffer))
        a23 = np.real(np.fft.ifftn(y * z * besseljBuffer))
        a33 = np.real(np.fft.ifftn(z * z * besseljBuffer))

        # compute eigenvalues
        l1, l2, l3 = eigenvaluefield33(a11, a12, a13, a22, a23, a33)

        # sort eigenvalues l1 >> l2 >> l3
        l1, l2, l3 = sort_eigenvalues(l1, l2, l3, absolute=opts['use_absolute'])

        # compute response
        feature = compute_response(l1, l2, rtype=opts['resp_type'])

        # compute output
        condition = (np.abs(feature) > np.abs(output_data))
        output_data[condition] = feature[condition]

    oof_response_image = sitk.GetImageFromArray(output_data)
    oof_response_image.SetOrigin(image.GetOrigin())
    oof_response_image.SetSpacing(image.GetSpacing())
    oof_response_image.SetDirection(image.GetDirection())

    return sitk.Cast(oof_response_image, sitk.sitkFloat32)


def ifft_shifted_coord_matrix(_size, _spacing):
    size = np.asarray(_size)
    spacing = np.asarray(_spacing)
    dim = len(_size)
    p = size // 2
    coord = [0]*dim

    for i in range(dim):
        # [half:end plus start:half] - half
        x, y = np.arange(p[i], size[i]), np.arange(p[i])
        a = np.concatenate((x, y)) - p[i]
        reshapepara = np.ones(dim, np.uint16)
        reshapepara[i] = size[i]
        A = np.reshape(a, reshapepara)
        repmatpara = np.copy(size)
        repmatpara[i] = 1
        coord[i] = np.tile(A, repmatpara).astype(np.double)

    return coord


def eigenvaluefield33(a11, a12, a13, a22, a23, a33):
    """
    Calculate the eigenvalues of massive 3x3 real symmetric matrices
    :param 3x3 symmetric matrix components
    :return: eigenvalues
    """
    eigenvalues = [0]*3
    epsilon = 1e-50

    b = a11 + epsilon
    d = a22 + epsilon
    j = a33 + epsilon

    c = - (a12**2 + a13**2 + a23**2 - b*d - d*j - j*b)
    d = - (b*d*j - a23**2*b - a12**2*j - a13**2*d + 2*a13*a12*a23)
    b = - a11 - a22 - a33 - 3*epsilon

    d += ( (2*b**3) - (9*b*c) ) / 27

    c = (b**2 / 3 - c)**3 / 27
    np.maximum(0, c, out=c)
    np.sqrt(c, out=c)

    j = c**(1/3)
    c += (c == 0)
    d = - d / 2 / c
    np.minimum(d, 1, out=d)
    np.maximum(d, -1, out=d)
    d = np.real( np.arccos(d) / 3)
    c = j*np.cos(d)
    d = j*np.sqrt(3)*np.sin(d)
    b = - b / 3

    j = - c - d + b
    d += - c + b
    b += 2*c

    return b.astype(np.single), j.astype(np.single), d.astype(np.single)


def sort_eigenvalues(l1, l2, l3, absolute=True):
    maxe, mine, mide = np.copy(l1), np.copy(l1), l1+l2+l3

    if absolute:
        maxe[np.abs(l2) > np.abs(maxe)] = l2[np.abs(l2) > np.abs(maxe)]
        mine[np.abs(l2) < np.abs(mine)] = l2[np.abs(l2) < np.abs(mine)]

        maxe[np.abs(l3) > np.abs(maxe)] = l3[np.abs(l3) > np.abs(maxe)]
        mine[np.abs(l3) < np.abs(mine)] = l3[np.abs(l3) < np.abs(mine)]

    else:
        maxe[l2 > np.abs(maxe)] = l2[l2 > np.abs(maxe)]
        mine[l2 < np.abs(mine)] = l2[l2 < np.abs(mine)]

        maxe[l3 > np.abs(maxe)] = l3[l3 > np.abs(maxe)]
        mine[l3 < np.abs(mine)] = l3[l3 < np.abs(mine)]

    mide -= maxe + mine

    return maxe, mide, mine


def compute_response(l1, l2, rtype=0):
    tmpfeature = 0
    if rtype == 0:
        tmpfeature = l1
    elif rtype == 1:
        tmpfeature = l1 + l2
    elif rtype == 2:
        tmpfeature = np.sqrt(np.maximum(0, l1*l2))
    elif rtype == 3:
        tmpfeature = np.sqrt(np.maximum(0,l1) * np.maximum(0,l2))
    elif rtype == 4:
        tmpfeature = np.maximum(l1, 0)
    elif rtype == 5:
        tmpfeature = np.maximum(l1+l2, 0)

    return tmpfeature