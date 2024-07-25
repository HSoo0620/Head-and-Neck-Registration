import pydicom
import numpy as np
import torch
from einops.einops import rearrange

def dvf_tensor2dcm(dvf_dcm_sample, dvf_flow, GridDimensions, dcm_metaInfo):
    '''
    dvf_dcm_sample : DVF DCM 파일 샘플. 해당 파일을 기본 틀로 새로운 DVF dcm 생성.
    dvf_flow : model's dvf output
        torch.tensor(B, W, H, D, 3(x,y,z))
    GridDimensions : 이미지 Shape
        list [w, h, d]
    dcm_metaInfo : dcm 이미지 데이터 meta data
        (voxel_spacing, initial_patient_position)
        voxel_spacing : [d, w, h] spacing
        initial_patient_position : [w, h, d] position
    '''
    # dvf_dcm = pydicom.read_file(dvf_dcm_sample)
    voxel_spacing, initial_patient_position = dcm_metaInfo
    # print(initial_patient_position)
    dvf_dcm = dvf_dcm_sample
    new_dvf_dcm = dvf_dcm
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImagePositionPatient = initial_patient_position # [-350.000000, -274.000000, -186.000000]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImageOrientationPatient = [1.000000, 0.000000, 0.000000,
                                                                                                                0.000000, 1.000000, 0.000000]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridDimensions = GridDimensions # [512, 512, 135]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridResolution = [voxel_spacing[1], voxel_spacing[2], voxel_spacing[0]] # [1.367188, 1.367188, 3.0]
    
    grid_resolution = new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridResolution
    grid_resolution = np.array(grid_resolution, dtype=np.float32)

    dvf_flow = dvf_flow[0] # w h d c(x,y,z)
    dvf_flow = rearrange(dvf_flow, 'w h d c -> d h w c')

    dvf_flow_patient = dvf_flow.cpu().numpy() * grid_resolution # d h w c(x,y,z)
    # dvf_flow_patient = dvf_flow.cpu().numpy() # d h w c(x,y,z)
    dvf_flow_patient = dvf_flow_patient.tobytes()
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].VectorGridData = dvf_flow_patient
    
    return new_dvf_dcm

def dvf_tensor2dcm_2(dvf_dcm_sample, dvf_flow, moving_dcm_metaInfo, fixed_dcm_metaInfo):
    '''
    dvf_dcm_sample : DVF DCM 파일 샘플. 해당 파일을 기본 틀로 새로운 DVF dcm 생성.
    dvf_flow : model's dvf output
        torch.tensor(B, W, H, D, 3(x,y,z))
    GridDimensions : 이미지 Shape
        list [w, h, d]
    dcm_metaInfo : dcm 이미지 데이터 meta data
        (voxel_spacing, initial_patient_position)
        voxel_spacing : [d, w, h] spacing
        initial_patient_position : [w, h, d] positionorg_moved_img
    '''
    # dvf_dcm = pydicom.read_file(dvf_dcm_sample)
    # voxel_spacing, initial_patient_position = moving_dcm_metaInfo
    # fixed_voxel_spacing, fixed_initial_patient_position = fixed_dcm_metaInfo

    moving_GridDimensions, moving_initial_patient_position, moving_voxel_spacing = moving_dcm_metaInfo
    fixed_GridDimensions, fixed_initial_patient_position, fixed_voxel_spacing = fixed_dcm_metaInfo

    # print('fixed dimensions: ', fixed_GridDimensions)
    # print('fixed origin: ', fixed_initial_patient_position)
    # print('fixed spacing: ', fixed_voxel_spacing)

    dvf_dcm = dvf_dcm_sample
    new_dvf_dcm = dvf_dcm
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImagePositionPatient = [fixed_initial_patient_position[0].item(),
                                                                                                                fixed_initial_patient_position[1].item(),
                                                                                                                fixed_initial_patient_position[2].item()] # e.g., [-350.000000, -274.000000, -186.000000]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImageOrientationPatient = [1.000000, 0.000000, 0.000000,
                                                                                                                0.000000, 1.000000, 0.000000]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridDimensions = [int(fixed_GridDimensions[0]),
                                                                                                          int(fixed_GridDimensions[1]),
                                                                                                          int(fixed_GridDimensions[2])] # e.g., [512, 512, 135]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridResolution = [fixed_voxel_spacing[0].item(), 
                                                                                                          fixed_voxel_spacing[1].item(),
                                                                                                          fixed_voxel_spacing[2].item()] # e.g., [1.367188, 1.367188, 3.0]
    
    dvf_flow = dvf_flow[0] # w h d c(x,y,z)
    dvf_flow = rearrange(dvf_flow, 'w h d c -> d h w c')

    dvf_flow_patient = dvf_flow.cpu().numpy().astype(np.float32) # d h w c(x,y,z)
    dvf_flow_patient = dvf_flow_patient.tobytes()
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].VectorGridData = dvf_flow_patient
    
    return new_dvf_dcm

def dvf_tensor2dcm_3(dvf_dcm_sample, dvf_flow, fixed_dcm_metaInfo, affine_matrix):
    '''
    dvf_dcm_sample : DVF DCM 파일 샘플. 해당 파일을 기본 틀로 새로운 DVF dcm 생성.
    dvf_flow : model's dvf output
        torch.tensor(B, W, H, D, 3(x,y,z))
    dcm_metaInfo : dcm 이미지 데이터 meta data
        (voxel_spacing, initial_patient_position)
        voxel_spacing : [d, w, h] spacing
        initial_patient_position : [w, h, d] positionorg_moved_img
    affine_matrix : 4x4 pre-deform affine matrix
    '''

    fixed_GridDimensions, fixed_initial_patient_position, fixed_voxel_spacing = fixed_dcm_metaInfo

    flatten_matrix = [affine_matrix[0][0].item(), affine_matrix[0][1].item(), affine_matrix[0][2].item(), affine_matrix[0][3].item(),
                      affine_matrix[1][0].item(), affine_matrix[1][1].item(), affine_matrix[1][2].item(), affine_matrix[1][3].item(),
                      affine_matrix[2][0].item(), affine_matrix[2][1].item(), affine_matrix[2][2].item(), affine_matrix[2][3].item(),
                      0, 0, 0, 1]


    dvf_dcm = dvf_dcm_sample
    new_dvf_dcm = dvf_dcm

    new_dvf_dcm.DeformableRegistrationSequence[0].PreDeformationMatrixRegistrationSequence[0].FrameOfReferenceTransformationMatrix = flatten_matrix
    print(fixed_initial_patient_position)
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImagePositionPatient = [fixed_initial_patient_position[0].item(),
                                                                                                                fixed_initial_patient_position[1].item(),
                                                                                                                fixed_initial_patient_position[2].item()] # e.g., [-350.000000, -274.000000, -186.000000]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImageOrientationPatient = [1.000000, 0.000000, 0.000000,
                                                                                                                0.000000, 1.000000, 0.000000]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridDimensions = [int(fixed_GridDimensions[0]),
                                                                                                          int(fixed_GridDimensions[1]),
                                                                                                          int(fixed_GridDimensions[2])] # e.g., [512, 512, 135]
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridResolution = [fixed_voxel_spacing[0].item(), 
                                                                                                          fixed_voxel_spacing[1].item(),
                                                                                                          fixed_voxel_spacing[2].item()] # e.g., [1.367188, 1.367188, 3.0]
    
    dvf_flow = dvf_flow[0] # w h d c(x,y,z)
    dvf_flow = rearrange(dvf_flow, 'w h d c -> d h w c')

    dvf_flow_patient = dvf_flow.cpu().numpy().astype(np.float32) # d h w c(x,y,z)
    dvf_flow_patient = dvf_flow_patient.tobytes()
    new_dvf_dcm.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].VectorGridData = dvf_flow_patient
    
    return new_dvf_dcm