import cv2
import numpy as np
import scipy.io as sio
from skimage import io
from faceutil import mesh
# from data import bfm, modelParam2Mesh, UVMap2Mesh
import matplotlib.pyplot as plt
from skimage import io, transform
from data import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm
from lookMatForVisual import  getKp3d


def showLandmark(image, kpt):
    image=plt.imread(image)
    kpt=np.load(kpt)
    kpt = np.round(kpt).astype(np.int32)
    image[kpt[:, 1], kpt[:, 0]] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    plt.imshow(image)
    plt.show()


def showLandmark2(image, kpt1, kpt2):
    image = plt.imread(image)
    kpt1 = np.load(kpt1)
    kpt2 = getKp3d(kpt2)
    kpt1 = np.round(kpt1).astype(np.int)
    kpt2 = np.round(kpt2).astype(np.int)
    kp1color=np.array([255, 0, 0])
    kp2color = np.array([0, 255, 0])
    image[kpt1[:, 1], kpt1[:, 0]] = kp1color
    image[kpt1[:, 1] + 1, kpt1[:, 0] + 1] = kp1color
    image[kpt1[:, 1] - 1, kpt1[:, 0] + 1] = kp1color
    image[kpt1[:, 1] - 1, kpt1[:, 0] - 1] = kp1color
    image[kpt1[:, 1] + 1, kpt1[:, 0] - 1] = kp1color

    image[kpt2[:, 1], kpt2[:, 0]] = kp2color
    image[kpt2[:, 1] + 1, kpt2[:, 0]] = kp2color
    image[kpt2[:, 1] - 1, kpt2[:, 0]] = kp2color
    image[kpt2[:, 1], kpt2[:, 0] + 1] = kp2color
    image[kpt2[:, 1], kpt2[:, 0] - 1] = kp2color

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def showGTLandmark(image_path):
    image = io.imread(image_path) / 255.0
    bfm_info = sio.loadmat(image_path.replace('jpg', 'mat'))
    if 'pt3d_68' in bfm_info.keys():
        kpt = bfm_info['pt3d_68'].T
    else:
        kpt = bfm_info['pt2d'].T
    showLandmark(image, kpt)

    mesh_info = bfm2Mesh(bfm_info, image.shape)

    kpt2 = mesh_info['vertices'][bfm.kpt_ind]
    showLandmark2(image, kpt, kpt2)
    return kpt, kpt2


def showImage(image, is_path=False):
    if is_path:
        img = io.imread(image) / 255.
        io.imshow(img)
        plt.show()
    else:
        io.imshow(image)
        plt.show()


def showMesh(mesh_info, init_img=None):
    height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
    width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    channel = 3
    if init_img is not None:
        [height, width, channel] = init_img.shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                           height, width, channel)
    if init_img is None:
        io.imshow(mesh_image)
        plt.show()
        getLandmark(mesh_image,)
    else:
        # plt.subplot(1, 3, 1)
        # plt.imshow(mesh_image)
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(init_img)
        # plt.axis('off')

        verify_img = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                               height, width, channel, BG=init_img)
        # plt.subplot(1, 3, 2)
        plt.imshow(verify_img)
        plt.axis('off')

        plt.show()


def showMesh2(mesh_info, init_img=None):
    height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
    width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    channel = 3
    if init_img is not None:
        [height, width, channel] = init_img.shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                           height, width, channel)
    if init_img is None:
        io.imshow(mesh_image)
        plt.show()
        getLandmark(mesh_image,)
    else:

        verify_img = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                               height, width, channel, BG=init_img)

        plt.imshow(verify_img)

        plt.show()


def show000(ipt, is_file=False, mode='image'):
    if mode == 'image':
        if is_file:
            # ipt is a path
            image = io.imread(ipt) / 255.
        else:
            image = ipt
        io.imshow(image)
        plt.show()
    elif mode == 'uvmap':
        # ipt should be [posmap texmap] or [posmap texmap image]
        assert (len(ipt) > 1)
        init_image = None
        if is_file:
            uv_position_map = np.load(ipt[0])
            uv_texture_map = io.imread(ipt[1])
            if len(ipt) > 2:
                init_image = io.imread(ipt[2]) / 255.
        else:
            uv_position_map = ipt[0]
            uv_texture_map = ipt[1]
            if len(ipt) > 2:
                init_image = ipt[2]

        mesh_info = UVmap2Mesh(uv_position_map=uv_position_map, uv_texture_map=uv_texture_map)
        showMesh(mesh_info, init_image)
    elif mode == 'mesh':
        if is_file:
            if len(ipt) == 2:
                mesh_info = sio.loadmat(ipt[0])
                init_image = io.imread(ipt[1]) / 255.
            else:
                mesh_info = sio.loadmat(ipt)
                init_image = None
        else:
            if len(ipt == 2):
                mesh_info = ipt[0]
                init_image = ipt[1]
            else:
                mesh_info = ipt
                init_image = None
        showMesh(mesh_info, init_image)

def show(ipt, texture=None,is_file=False, mode='image'):
    if mode == 'image':
        if is_file:
            # ipt is a path
            image = io.imread(ipt) / 255.
        else:
            image = ipt
        io.imshow(image)
        plt.show()
    elif mode == 'uvmap':
        # ipt should be [posmap texmap] or [posmap texmap image]
        assert (len(ipt) > 1)
        init_image = None
        if is_file:
            uv_position_map = np.load(ipt[0])
            uv_texture_map = io.imread(ipt[1]) / 255.
            if len(ipt) > 2:
                init_image = io.imread(ipt[2]) / 255.
        else:
            uv_position_map = ipt[0]
            uv_texture_map = ipt[1]
            if len(ipt) > 2:
                init_image = ipt[2]

        mesh_info = UVmap2Mesh(uv_position_map=uv_position_map, uv_texture_map=uv_texture_map)
        showMesh(mesh_info, init_image)
    elif mode == 'white':
        # ipt should be [posmap texmap] or [posmap texmap image]
        texture = io.imread(texture)/255
        init_image = io.imread(ipt[1]) / 255.
        init_image=np.array(init_image,dtype=np.float32)
        if is_file:
            uv_position_map = np.load(ipt[0])


            if len(ipt) > 2:
                init_image = io.imread(ipt[2]) / 255.
        else:
            uv_position_map = ipt[0]

            if len(ipt) > 2:
                init_image = ipt[2]

        # ↓没纹理的
        mesh_info = UVmap2Mesh(uv_position_map=uv_position_map,uv_texture_map=cv2.cvtColor((uv_position_map[:,:,2])**2/5/10000,cv2.COLOR_GRAY2RGB))
        # mesh_info = UVmap2Mesh(uv_position_map=uv_position_map,
        #                        uv_texture_map=None)

        # ↓有纹理的
        # texture=cv2.cvtColor(texture[:,:,2],cv2.COLOR_GRAY2RGB)
        # mesh_info = UVmap2Mesh(uv_position_map=uv_position_map,uv_texture_map=texture)

        showMesh(mesh_info, init_image)
    elif mode == 'mesh':
        if is_file:
            if len(ipt) == 2:
                mesh_info = sio.loadmat(ipt[0])
                init_image = io.imread(ipt[1]) / 255.
            else:
                mesh_info = sio.loadmat(ipt)
                init_image = None
        else:
            if len(ipt == 2):
                mesh_info = ipt[0]
                init_image = ipt[1]
            else:
                mesh_info = ipt
                init_image = None
        showMesh(mesh_info, init_image)


if __name__ == "__main__":
    # pass
    # showUVMap('data/images/AFLW2000-out/image00002/image00002_uv_posmap.npy', None,
    #           # 'data/images/AFLW2000-output/image00002/image00002_uv_texture_map.jpg',
    #           'data/images/AFLW2000-out/image00002/image00002_init.jpg', True)
    # show(['data/images/AFLW2000-crop-offset/image00002/image00002_cropped_uv_posmap.npy',
    #       'data/images/AFLW2000-crop/image00002/image00002_uv_texture_map.jpg',
    #       'data/images/AFLW2000-crop-offset/image00002/image00002_cropped.jpg'], is_file=True, mode='uvmap')
    # show(['Q:/GitHub/PRNet-PyTorch-BU/data/images/AFLW2000-crop/image00002/image00002_cropped_uv_posmap.npy',
    #       'Q:/GitHub/PRNet-PyTorch-BU/data/images/AFLW2000-crop/image00002/image00002_uv_texture_map.jpg'], is_file=True, mode='uvmap')

    # #画一个landmark
    # showLandmark('Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll030\image00040\image00040_cropped.jpg',
    #              'Q:\GitHub\PRNetOUT\8rotSEtopk\Yaw030Roll030\image00014KP.npy')



    # # 密集人脸↓
    # #正确的
    # show(['Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll030\image00013\image00013_uv_posmap.npy',
    #       'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll030\image00013\image00013_uv_texture_map.jpg'], is_file=True, mode='uvmap')

    angle = '030'
    imageID = '4358'
    imageRank = '3990'
    model = '8rotSEtopk'

    # ground truth展示
    show(['Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-crop\\'+'\image0' + imageID + '\image0' + imageID+ '_cropped_uv_posmap.npy',
          'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-crop' + '\image0' + imageID + '\image0' + imageID + '_cropped.jpg'],
         'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-crop' + '\image0' + imageID + '\image0' + imageID + '_uv_texture_map.jpg',
         is_file=True, mode='white')

    #非Yaw030Roll分类 All
    # show(['Q:\GitHub\PRNetOUT\\' + model + '\\All'  + '\image000' + imageRank + '.npy',
    #       'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-crop'  + '\image0' + imageID + '\image0' + imageID + '_cropped.jpg'],
    #      'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-crop'  + '\image0' + imageID + '\image0' + imageID + '_uv_texture_map.jpg',
    #      is_file=True, mode='white')


    #All的双关键点
    # showLandmark2(
    #     'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-crop'  + '\image0' + imageID + '\image0' + imageID + '_cropped.jpg',
    #     'Q:\GitHub\PRNetOUT\\' + model + '\\All'  + '\image000' + imageRank + 'KP.npy',
    #     'Q:\GitHub\PRNet-PyTorch-BU\data\images\\AFLW2000-crop' + '\image0' + imageID + '\image0' + imageID + '_bbox_info.mat')

    # 030的双关键点
    # showLandmark2('Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll'+angle+'\image0'+imageID+'\image0'+imageID+'_cropped.jpg',
    #               'Q:\GitHub\PRNetOUT\\'+model+'\Yaw030Roll'+angle+'\image000'+imageRank+'KP.npy',
    #               'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll'+angle+'\image0'+imageID+'\image0'+imageID+'_bbox_info.mat')

    #没texture
    # show(['Q:\GitHub\PRNetOUT\\' + model + '\\Yaw030Roll' + angle + '\image000' + imageRank + '.npy',
    #       'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll' + angle + '\image0' + imageID + '\image0' + imageID + '_cropped.jpg'],
    #      None,
    #      is_file=True, mode='mesh')

    # # 密集人脸
    # show(['Q:\GitHub\PRNetOUT\\'+model+'\\Yaw030Roll'+angle+'\image000'+imageRank+'.npy',
    #       'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll'+angle+'\image0'+imageID+'\image0'+imageID+'_cropped.jpg'],
    #      'Q:\GitHub\PRNet-PyTorch-BU\data\images\AFLW2000-cropYaw030Roll' + angle + '\image0' + imageID + '\image0' + imageID + '_uv_texture_map.jpg',
    #      is_file=True, mode='white')

