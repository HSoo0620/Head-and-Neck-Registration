# Coded version of DICOM file 'dvf/dro_1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235184.437123.dcm'
# Produced by pydicom codify utility script
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

def dvf_sample():
    '''
    Output : DVF dcm Dataset.file_meta
    '''
    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.3'
    file_meta.MediaStorageSOPInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235184.437123'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationClassUID = '1.2.276.0.7230010.3.0.3.6.6'
    file_meta.ImplementationVersionName = 'OFFIS_DCMTK_366'

    # Main data elements
    ds = Dataset()
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.InstanceCreationDate = '20230224'
    ds.InstanceCreationTime = '193942'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.3'
    ds.SOPInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235184.437123'
    ds.StudyDate = '20230224'
    ds.SeriesDate = ''
    ds.ContentDate = '20230224'
    ds.StudyTime = '193942'
    ds.SeriesTime = ''
    ds.ContentTime = '193942'
    ds.AccessionNumber = ''
    ds.Modality = 'REG'
    ds.Manufacturer = 'Plastimatch'
    ds.ReferringPhysicianName = ''
    ds.StationName = ''
    ds.StudyDescription = ''
    ds.SeriesDescription = ''
    ds.OperatorsName = ''
    ds.ManufacturerModelName = 'Plastimatch'

    # Referenced Series Sequence
    refd_series_sequence = Sequence()
    ds.ReferencedSeriesSequence = refd_series_sequence

    # Referenced Series Sequence: Referenced Series 1
    refd_series1 = Dataset()

    # Referenced Instance Sequence
    refd_instance_sequence = Sequence()
    refd_series1.ReferencedInstanceSequence = refd_instance_sequence

    # Referenced Instance Sequence: Referenced Instance 1
    refd_instance1 = Dataset()
    refd_instance1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    refd_instance1.ReferencedSOPInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235182.436561'
    refd_instance_sequence.append(refd_instance1)

    refd_series1.SeriesInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235182.436558'
    refd_series_sequence.append(refd_series1)

    # Referenced Series Sequence: Referenced Series 2
    refd_series2 = Dataset()

    # Referenced Instance Sequence
    refd_instance_sequence = Sequence()
    refd_series2.ReferencedInstanceSequence = refd_instance_sequence

    # Referenced Instance Sequence: Referenced Instance 1
    refd_instance1 = Dataset()
    refd_instance1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    refd_instance1.ReferencedSOPInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235183.436852'
    refd_instance_sequence.append(refd_instance1)

    refd_series2.SeriesInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235183.436849'
    refd_series_sequence.append(refd_series2)

    ds.PatientName = 'ANONYMOUS'
    ds.PatientID = 'PL618999465318312'
    ds.PatientBirthDate = ''
    ds.PatientSex = 'O'
    ds.DeviceSerialNumber = ''
    ds.SoftwareVersions = '436aebba'
    ds.PatientPosition = 'HFS'
    ds.StudyInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235183.436831'
    ds.SeriesInstanceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235184.437122'
    ds.StudyID = ''
    ds.SeriesNumber = None
    ds.FrameOfReferenceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235183.436832'

    # Deformable Registration Sequence
    deformable_registration_sequence = Sequence()
    ds.DeformableRegistrationSequence = deformable_registration_sequence

    # Deformable Registration Sequence: Deformable Registration 1
    deformable_registration1 = Dataset()
    deformable_registration1.SourceFrameOfReferenceUID = '1.2.826.0.1.3680043.8.274.1.1.0.30692.1677235182.436541'

    # Deformable Registration Grid Sequence
    deformable_registration_grid_sequence = Sequence()
    deformable_registration1.DeformableRegistrationGridSequence = deformable_registration_grid_sequence

    # Deformable Registration Grid Sequence: Deformable Registration Grid 1
    deformable_registration_grid1 = Dataset()
    deformable_registration_grid1.ImagePositionPatient = [-350.000000, -274.000000, -186.000000]
    deformable_registration_grid1.ImageOrientationPatient = [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
    deformable_registration_grid1.GridDimensions = [512, 512, 135]
    deformable_registration_grid1.GridResolution = [1.367188, 1.367188, 3.0]
    deformable_registration_grid1.VectorGridData = None # XXX Array of 424673280 bytes excluded
    deformable_registration_grid_sequence.append(deformable_registration_grid1)


    # Pre Deformation Matrix Registration Sequence
    pre_deformation_matrix_registration_sequence = Sequence()
    deformable_registration1.PreDeformationMatrixRegistrationSequence = pre_deformation_matrix_registration_sequence

    # Pre Deformation Matrix Registration Sequence: Pre Deformation Matrix Registration 1
    pre_deformation_matrix_registration1 = Dataset()
    pre_deformation_matrix_registration1.FrameOfReferenceTransformationMatrixType = 'RIGID'
    pre_deformation_matrix_registration1.FrameOfReferenceTransformationMatrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    pre_deformation_matrix_registration_sequence.append(pre_deformation_matrix_registration1)


    # Post Deformation Matrix Registration Sequence
    post_deformation_matrix_registration_sequence = Sequence()
    deformable_registration1.PostDeformationMatrixRegistrationSequence = post_deformation_matrix_registration_sequence

    # Post Deformation Matrix Registration Sequence: Post Deformation Matrix Registration 1
    post_deformation_matrix_registration1 = Dataset()
    post_deformation_matrix_registration1.FrameOfReferenceTransformationMatrixType = 'RIGID'
    post_deformation_matrix_registration1.FrameOfReferenceTransformationMatrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    post_deformation_matrix_registration_sequence.append(post_deformation_matrix_registration1)


    # Registration Type Code Sequence
    registration_type_code_sequence = Sequence()
    deformable_registration1.RegistrationTypeCodeSequence = registration_type_code_sequence

    # Registration Type Code Sequence: Registration Type Code 1
    registration_type_code1 = Dataset()
    registration_type_code_sequence.append(registration_type_code1)
    deformable_registration_sequence.append(deformable_registration1)

    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    # ds.save_as(r'dvf/save.dcm', write_like_original=False)
    return ds