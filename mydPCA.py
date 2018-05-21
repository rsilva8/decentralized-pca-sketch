
import numpy as np
from base_PCA import base_PCA


def dPCA(site_list, num_PC, mean_removal=None, subject_level_PCA=True, subject_level_num_PC=120):
    """ Decentralized PCA

    Principal component analysis of decentralized data.
    Accounts for local site mean removal and subject-level whitening.
    """
    R = 5 * num_PC # Intermediate dimension is 5 times the desired final dimension
    for ii in site_list:
        # Retrieve reduced local site data
        reduced_data_site[ii] = local_PCA(site=ii,
                                          num_PC=R,
                                          mean_removal=mean_removal,
                                          subject_level_PCA=subject_level_PCA,
                                          subject_level_num_PC=subject_level_num_PC)
    
    PC_global, projM_global, bkprojM_global = base_PCA(np.hstack(reduced_data_site),
                                                       num_PC=num_PC,
                                                       axis=1,
                                                       whitening=False)
    return PC_global, projM_global, bkprojM_global


def local_PCA(site, num_PC, mean_removal=None, subject_level_PCA=True, subject_level_num_PC=120):
    """ Local PCA

    Local principal component analysis method for dPCA.
    Accounts for mean removal and subject-level whitening.
    """
    subject_list = load_site_subject_list(site)
    for mm in subject_list:
        raw_subject = load_subject_data(mm)

        if mean_removal:
            axis, mean_values = mean_removal # mean_removal is a tuple
            if axis == 0:
                # Remove column means
                # Ignore contents of mean_values
                raw_subject = raw_subject - np.mean(raw_subject)
            elif axis == 1:
                # Remove row means
                # mean_values computed in decentralized fashion elsewhere
                raw_subject = raw_subject - mean_values[:,None]

        if subject_level_PCA:
            # This is subject level PCA with whitening
            data_subject[mm], projM_subject[mm], bkprojM_subject[mm] = base_PCA(raw_subject,
                                                                                num_PC=subject_level_num_PC,
                                                                                axis=1,
                                                                                whitening=True)
            save_subject_projection_matrices(mm, projM_subject, bkprojM_subject)
        else:
            data_subject[mm] = raw_subject

    reduced_data, _, _ = base_PCA(np.hstack(data_subject),
                                  num_PC=num_PC,
                                  axis = 1,
                                  whitening=False)
    return reduced_data


def djICA_main(list_of_sites):
    # Rough example of how to use dPCA for djICA
    final_num_PC = 100
    row_mean_values = decentralized_row_means(list_of_sites)
    axis = 1
    mean_removal = (axis, row_mean_values)
    global_PC, _, _ = dPCA(list_of_sites,
                           num_PC=final_num_PC,
                           mean_removal=mean_removal,
                           subject_level_PCA=False)
    decentralized_data_reducer(global_PC)
    djICA()

def groupICA_mean(list_of_sites):
    # Rough example of how to use dPCA for group ICA
    final_num_PC = 100
    axis = 0
    mean_removal = (axis, None)
    PC_global, projM_global, bkprojM_global = dPCA(list_of_sites,
                                                   num_PC=final_num_PC,
                                                   mean_removal=mean_removal,
                                                   subject_level_PCA=True,
                                                   subject_level_num_PC=120)
    save_dPCA_results(PC_global, projM_global, bkprojM_global)
    ICA(PC_global.T)
