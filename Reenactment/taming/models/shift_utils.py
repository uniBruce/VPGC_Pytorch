import io
import os
import math
import torch
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _get_inverse_affine_matrix(center, angle, translate, scale, shear=None, img_size=[256, 192]):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    matrix = np.array(matrix).reshape(2, 3)
    matrix[0, 2] /= img_size[0]
    matrix[1, 2] /= img_size[1]

    return matrix


def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


def compute_affine_matrices(batch_size, img_size, degrees=[-6, 6], translate=[0.04, 0.04], scale_ranges=[0.98, 1.02]): # we double degree and translate
    center = [img_size[0] * 0.5, img_size[1] * 0.5]

    angle, translations, scale, shear = get_params(degrees, translate, scale_ranges, None, img_size)
    affine_mat = _get_inverse_affine_matrix(center, angle, translate, scale, shear, img_size)
    affine_mat = torch.from_numpy(affine_mat).unsqueeze(0).float()
    # inv_affine_mat = _get_inverse_affine_matrix(center, -angle, -translate, 1./scale, shear)
    # inv_affine_mat = torch.from_numpy(inv_affine_mat).unsqueeze(0).float()

    for i in range(1, batch_size):
        angle, translations, scale, shear = get_params(degrees, translate, scale_ranges, None, img_size)
        affine_mat_ = _get_inverse_affine_matrix(center, angle, translate, scale, shear, img_size)
        affine_mat_ = torch.from_numpy(affine_mat_).unsqueeze(0).float()
        # inv_affine_mat_ = _get_inverse_affine_matrix(center, -angle, -translate, 1./scale, shear)
        # inv_affine_mat_ = torch.from_numpy(inv_affine_mat_).unsqueeze(0).float()
        affine_mat = torch.cat([affine_mat, affine_mat_], 0)
        # inv_affine_mat = torch.cat([inv_affine_mat, inv_affine_mat_], 0)

    return affine_mat#, inv_affine_mat

