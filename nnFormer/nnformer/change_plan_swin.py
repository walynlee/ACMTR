from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task id")
    args = parser.parse_args()

    if args.task=='2':
        input_file = '../DATASET/nnFormer_preprocessed/Task02_Synapse/nnFormerPlansv2.1_plans_3D.pkl'
        output_file = '../DATASET/nnFormer_preprocessed/Task02_Synapse/nnFormerPlansv2.1_Synapse_plans_3D.pkl'
        a = load_pickle(input_file)
        
        a['plans_per_stage'][0]['patch_size']=np.array([64,128,128])
        a['plans_per_stage'][0]['pool_op_kernel_sizes']=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
        a['plans_per_stage'][0]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        save_pickle(a, output_file)
        
        # Here is to change the dataset division
        split_file=input_file.replace('nnFormerPlansv2.1_plans_3D','splits_final')
        b = load_pickle(split_file)
        b[0]['train']=np.array(['img0006','img0007' ,'img0009', 'img0010', 'img0021' ,'img0023' ,'img0024','img0026' ,'img0027' ,'img0031', 'img0033' ,'img0034' \
                                ,'img0039', 'img0040','img0005', 'img0028', 'img0030', 'img0037'])
        b[0]['val']=np.array(['img0001', 'img0002', 'img0003', 'img0004', 'img0008', 'img0022','img0025', 'img0029', 'img0032', 'img0035', 'img0036', 'img0038'])
        save_pickle(b,split_file)

    elif args.task=='11':
        input_file = 'D:/walyn/ubuntu_copy/DATASET/nnFormer_preprocessed/Task11_PelvicTumour/nnFormerPlansv2.1_plans_3D.pkl'
        output_file = 'D:/walyn/ubuntu_copy/DATASET/nnFormer_preprocessed/Task11_PelvicTumour/nnFormerPlansv2.1_PelvicTumour_plans_3D.pkl'
        a = load_pickle(input_file)
        
        a['plans_per_stage'][0]['patch_size']=np.array([96,128,128])
        a['plans_per_stage'][0]['pool_op_kernel_sizes']=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
        a['plans_per_stage'][0]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        save_pickle(a, output_file)
        
        # Here is to change the dataset division
        split_file=input_file.replace('nnFormerPlansv2.1_plans_3D','splits_final')
        b = load_pickle(split_file)
        #93 例
        """b[0]['train']=np.array(['train_2855.nii.gz', 'train_2973.nii.gz', 'train_3036.nii.gz', 'train_3051.nii.gz', 
                'train_3070.nii.gz', 'train_3093.nii.gz', 'train_3212.nii.gz', 'train_3214.nii.gz', 'train_3251.nii.gz', 
                'train_3266.nii.gz', 'train_3269.nii.gz', 'train_3340.nii.gz', 'train_3341.nii.gz', 'train_3364.nii.gz', 
                'train_3443.nii.gz', 'train_3451.nii.gz', 'train_3454.nii.gz', 'train_3542.nii.gz', 'train_3547.nii.gz', 
                'train_3548.nii.gz', 'train_3560.nii.gz', 'train_3574.nii.gz', 'train_3620.nii.gz', 'train_3627.nii.gz', 
                'train_3636.nii.gz', 'train_3655.nii.gz', 'train_3718.nii.gz', 'train_3725.nii.gz', 'train_3816.nii.gz', 
                'train_3825.nii.gz', 'train_3827.nii.gz', 'train_3964.nii.gz', 'train_3970.nii.gz', 'train_4105.nii.gz', 
                'train_4148.nii.gz', 'train_4265.nii.gz', 'train_4363.nii.gz', 'train_4383.nii.gz', 'train_4485.nii.gz', 
                'train_4486.nii.gz', 'train_4487.nii.gz', 'train_4505.nii.gz', 'train_4540.nii.gz', 'train_4671.nii.gz', 
                'train_4757.nii.gz', 'train_4767.nii.gz', 'train_4802.nii.gz', 'train_4826.nii.gz', 'train_4995.nii.gz', 
                'train_5058.nii.gz', 'train_5076.nii.gz', 'train_5155.nii.gz', 'train_5162.nii.gz', 'train_5339.nii.gz', 
                'train_5373.nii.gz', 'train_5550.nii.gz', 'train_5551.nii.gz', 'train_5621.nii.gz', 'train_5726.nii.gz', 
                'train_5783.nii.gz', 'train_5793.nii.gz', 'train_5801.nii.gz', 'train_5849.nii.gz', 'train_5864.nii.gz', 
                'train_5897.nii.gz', 'train_5906.nii.gz', 'train_6077.nii.gz', 'train_6172.nii.gz', 'train_6285.nii.gz', 
                'train_6378.nii.gz', 'train_6494.nii.gz', 'train_6890.nii.gz', 'train_6933.nii.gz', 'train_6937.nii.gz', ])
        b[0]['val']=np.array(['train_6995.nii.gz', 'train_7050.nii.gz', 'train_7114.nii.gz', 'train_7149.nii.gz', 'train_7234.nii.gz', 
            'train_7327.nii.gz', 'train_7418.nii.gz', 'train_7498.nii.gz', 'train_7515.nii.gz', 'train_7546.nii.gz', 
            'train_7572.nii.gz', 'train_7625.nii.gz', 'train_7662.nii.gz', 'train_7663.nii.gz', 'train_7664.nii.gz', 
            'train_7756.nii.gz', 'train_7778.nii.gz', 'train_7997.nii.gz', 'train_8297.nii.gz']
        )"""
        #930 例
        b[0]['train']=np.array(['train_2855', 'train_2855_0', 'train_2855_1', 'train_2855_2', 'train_2855_3', 'train_2855_4', 'train_2855_5', 'train_2855_6', 'train_2855_7', 'train_2855_8', 'train_2973', 'train_2973_0', 'train_2973_1', 'train_2973_2', 'train_2973_3', 'train_2973_4', 'train_2973_5', 'train_2973_6', 'train_2973_7', 'train_2973_8', 'train_3036', 'train_3036_0', 'train_3036_1', 'train_3036_2', 'train_3036_3', 'train_3036_4', 'train_3036_5', 'train_3036_6', 'train_3036_7', 'train_3036_8', 'train_3070', 'train_3070_0', 'train_3070_1', 'train_3070_2', 'train_3070_3', 'train_3070_4', 'train_3070_5', 'train_3070_6', 'train_3070_7', 'train_3070_8', 'train_3093', 'train_3093_0', 'train_3093_1', 'train_3093_2', 'train_3093_3', 'train_3093_4', 'train_3093_5', 'train_3093_6', 'train_3093_7', 'train_3093_8', 'train_3212', 'train_3212_0', 'train_3212_1', 'train_3212_2', 'train_3212_3', 'train_3212_4', 'train_3212_5', 'train_3212_6', 'train_3212_7', 'train_3212_8', 'train_3214', 'train_3214_0', 'train_3214_1', 'train_3214_2', 'train_3214_3', 'train_3214_4', 'train_3214_5', 'train_3214_6', 'train_3214_7', 'train_3214_8', 'train_3251', 'train_3251_0', 'train_3251_1', 'train_3251_2', 'train_3251_3', 'train_3251_4', 'train_3251_5', 'train_3251_6', 'train_3251_7', 'train_3251_8', 'train_3266', 'train_3266_0', 'train_3266_1', 'train_3266_2', 'train_3266_3', 'train_3266_4', 'train_3266_5', 'train_3266_6', 'train_3266_7', 'train_3266_8', 'train_3269', 'train_3269_0', 'train_3269_1', 'train_3269_2', 'train_3269_3', 'train_3269_4', 'train_3269_5', 'train_3269_6', 'train_3269_7', 'train_3269_8', 'train_3340', 'train_3340_0', 'train_3340_1', 'train_3340_2', 'train_3340_3', 'train_3340_4', 'train_3340_5', 'train_3340_6', 'train_3340_7', 'train_3340_8', 'train_3341', 'train_3341_0', 'train_3341_1', 'train_3341_2', 'train_3341_3', 'train_3341_4', 'train_3341_5', 'train_3341_6', 'train_3341_7', 'train_3341_8', 'train_3364', 'train_3364_0', 'train_3364_1', 'train_3364_2', 'train_3364_3', 'train_3364_4', 'train_3364_5', 'train_3364_6', 'train_3364_7', 'train_3364_8', 'train_3443', 'train_3443_0', 'train_3443_1', 'train_3443_2', 'train_3443_3', 'train_3443_4', 'train_3443_5', 'train_3443_6', 'train_3443_7', 'train_3443_8', 'train_3451', 'train_3451_0', 'train_3451_1', 'train_3451_2', 'train_3451_3', 'train_3451_4', 'train_3451_5', 'train_3451_6', 'train_3451_7', 'train_3451_8', 'train_3454', 'train_3454_0', 'train_3454_1', 'train_3454_2', 'train_3454_3', 'train_3454_4', 'train_3454_5', 'train_3454_6', 'train_3454_7', 'train_3454_8', 'train_3542', 'train_3542_0', 'train_3542_1', 'train_3542_2', 'train_3542_3', 'train_3542_4', 'train_3542_5', 'train_3542_6', 'train_3542_7', 'train_3542_8', 'train_3547', 'train_3547_0', 'train_3547_1', 'train_3547_2', 'train_3547_3', 'train_3547_4', 'train_3547_5', 'train_3547_6', 'train_3547_7', 'train_3547_8', 'train_3548', 'train_3548_0', 'train_3548_1', 'train_3548_2', 'train_3548_3', 'train_3548_4', 'train_3548_5', 'train_3548_6', 'train_3548_7', 'train_3548_8', 'train_3560', 'train_3560_0', 'train_3560_1', 'train_3560_2', 'train_3560_3', 'train_3560_4', 'train_3560_5', 'train_3560_6', 'train_3560_7', 'train_3560_8', 'train_3574', 'train_3574_0', 'train_3574_1', 'train_3574_2', 'train_3574_3', 'train_3574_4', 'train_3574_5', 'train_3574_6', 'train_3574_7', 'train_3574_8', 'train_3620', 'train_3620_0', 'train_3620_1', 'train_3620_2', 'train_3620_3', 'train_3620_4', 'train_3620_5', 'train_3620_6', 'train_3620_7', 'train_3620_8', 'train_3627', 'train_3627_0', 'train_3627_1', 'train_3627_2', 'train_3627_3', 'train_3627_4', 'train_3627_5', 'train_3627_6', 'train_3627_7', 'train_3627_8', 'train_3636', 'train_3636_0', 'train_3636_1', 'train_3636_2', 'train_3636_3', 'train_3636_4', 'train_3636_5', 'train_3636_6', 'train_3636_7', 'train_3636_8', 'train_3655', 'train_3655_0', 'train_3655_1', 'train_3655_2', 'train_3655_3', 'train_3655_4', 'train_3655_5', 'train_3655_6', 'train_3655_7', 'train_3655_8', 'train_3718', 'train_3718_0', 'train_3718_1', 'train_3718_2', 'train_3718_3', 'train_3718_4', 'train_3718_5', 'train_3718_6', 'train_3718_7', 'train_3718_8', 'train_3725', 'train_3725_0', 'train_3725_1', 'train_3725_2', 'train_3725_3', 'train_3725_4', 'train_3725_5', 'train_3725_6', 'train_3725_7', 'train_3725_8', 'train_3816', 'train_3816_0', 'train_3816_1', 'train_3816_2', 'train_3816_3', 'train_3816_4', 'train_3816_5', 'train_3816_6', 'train_3816_7', 'train_3816_8', 'train_3825', 'train_3825_0', 'train_3825_1', 'train_3825_2', 'train_3825_3', 'train_3825_4', 'train_3825_5', 'train_3825_6', 'train_3825_7', 'train_3825_8', 'train_3827', 'train_3827_0', 'train_3827_1', 'train_3827_2', 'train_3827_3', 'train_3827_4', 'train_3827_5', 'train_3827_6', 'train_3827_7', 'train_3827_8', 'train_3964', 'train_3964_0', 'train_3964_1', 'train_3964_2', 'train_3964_3', 'train_3964_4', 'train_3964_5', 'train_3964_6', 'train_3964_7', 'train_3964_8', 'train_3970', 'train_3970_0', 'train_3970_1', 'train_3970_2', 'train_3970_3', 'train_3970_4', 'train_3970_5', 'train_3970_6', 'train_3970_7', 'train_3970_8', 'train_4105', 'train_4105_0', 'train_4105_1', 'train_4105_2', 'train_4105_3', 'train_4105_4', 'train_4105_5', 'train_4105_6', 'train_4105_7', 'train_4105_8', 'train_4148', 'train_4148_0', 'train_4148_1', 'train_4148_2', 'train_4148_3', 'train_4148_4', 'train_4148_5', 'train_4148_6', 'train_4148_7', 'train_4148_8', 'train_4265', 'train_4265_0', 'train_4265_1', 'train_4265_2', 'train_4265_3', 'train_4265_4', 'train_4265_5', 'train_4265_6', 'train_4265_7', 'train_4265_8', 'train_4363', 'train_4363_0', 'train_4363_1', 'train_4363_2', 'train_4363_3', 'train_4363_4', 'train_4363_5', 'train_4363_6', 'train_4363_7', 'train_4363_8', 'train_4383', 'train_4383_0', 'train_4383_1', 'train_4383_2', 'train_4383_3', 'train_4383_4', 'train_4383_5', 'train_4383_6', 'train_4383_7', 'train_4383_8', 'train_4485', 'train_4485_0', 'train_4485_1', 'train_4485_2', 'train_4485_3', 'train_4485_4', 'train_4485_5', 'train_4485_6', 'train_4485_7', 'train_4485_8', 'train_4486', 'train_4486_0', 'train_4486_1', 'train_4486_2', 'train_4486_3', 'train_4486_4', 'train_4486_5', 'train_4486_6', 'train_4486_7', 'train_4486_8', 'train_4487', 'train_4487_0', 'train_4487_1', 'train_4487_2', 'train_4487_3', 'train_4487_4', 'train_4487_5', 'train_4487_6', 'train_4487_7', 'train_4487_8', 'train_5155', 'train_5155_0', 'train_5155_1', 'train_5155_2', 'train_5155_3', 'train_5155_4', 'train_5155_5', 'train_5155_6', 'train_5155_7', 'train_5155_8', 'train_5162', 'train_5162_0', 'train_5162_1', 'train_5162_2', 'train_5162_3', 'train_5162_4', 'train_5162_5', 'train_5162_6', 'train_5162_7', 'train_5162_8', 'train_5339', 'train_5339_0', 'train_5339_1', 'train_5339_2', 'train_5339_3', 'train_5339_4', 'train_5339_5', 'train_5339_6', 'train_5339_7', 'train_5339_8', 'train_5373', 'train_5373_0', 'train_5373_1', 'train_5373_2', 'train_5373_3', 'train_5373_4', 'train_5373_5', 'train_5373_6', 'train_5373_7', 'train_5373_8', 'train_5550', 'train_5550_0', 'train_5550_1', 'train_5550_2', 'train_5550_3', 'train_5550_4', 'train_5550_5', 'train_5550_6', 'train_5550_7', 'train_5550_8', 'train_5551', 'train_5551_0', 'train_5551_1', 'train_5551_2', 'train_5551_3', 'train_5551_4', 'train_5551_5', 'train_5551_6', 'train_5551_7', 'train_5551_8', 'train_5621', 'train_5621_0', 'train_5621_1', 'train_5621_2', 'train_5621_3', 'train_5621_4', 'train_5621_5', 'train_5621_6', 'train_5621_7', 'train_5621_8', 'train_5726', 'train_5726_0', 'train_5726_1', 'train_5726_2', 'train_5726_3', 'train_5726_4', 'train_5726_5', 'train_5726_6', 'train_5726_7', 'train_5726_8', 'train_5783', 'train_5783_0', 'train_5783_1', 'train_5783_2', 'train_5783_3', 'train_5783_4', 'train_5783_5', 'train_5783_6', 'train_5783_7', 'train_5783_8', 'train_5793', 'train_5793_0', 'train_5793_1', 'train_5793_2', 'train_5793_3', 'train_5793_4', 'train_5793_5', 'train_5793_6', 'train_5793_7', 'train_5793_8', 'train_5801', 'train_5801_0', 'train_5801_1', 'train_5801_2', 'train_5801_3', 'train_5801_4', 'train_5801_5', 'train_5801_6', 'train_5801_7', 'train_5801_8', 'train_5849', 'train_5849_0', 'train_5849_1', 'train_5849_2', 'train_5849_3', 'train_5849_4', 'train_5849_5', 'train_5849_6', 'train_5849_7', 'train_5849_8', 'train_5864', 'train_5864_0', 'train_5864_1', 'train_5864_2', 'train_5864_3', 'train_5864_4', 'train_5864_5', 'train_5864_6', 'train_5864_7', 'train_5864_8', 'train_5897', 'train_5897_0', 'train_5897_1', 'train_5897_2', 'train_5897_3', 'train_5897_4', 'train_5897_5', 'train_5897_6', 'train_5897_7', 'train_5897_8', 'train_5906', 'train_5906_0', 'train_5906_1', 'train_5906_2', 'train_5906_3', 'train_5906_4', 'train_5906_5', 'train_5906_6', 'train_5906_7', 'train_5906_8'])
        b[0]['val']=np.array(['train_6077', 'train_6172', 'train_6285', 'train_6378', 'train_6494', 'train_6890', 'train_6933', 'train_6937', 'train_6995', 'train_7050', 'train_7114', 'train_7149', 'train_7234', 'train_7327', 'train_7418', 'train_7498', 'train_7515', 'train_7546', 'train_7572', 'train_7625', 'train_7662', 'train_7663', 'train_7664', 'train_7756', 'train_7778', 'train_7997', 'train_8297'])

        b[1]['train'] = np.array(['train_2855', 'train_2855_0', 'train_2855_1', 'train_2855_2', 'train_2855_3', 'train_2855_4', 'train_2855_5', 'train_2855_6', 'train_2855_7', 'train_2855_8', 'train_2973', 'train_2973_0', 'train_2973_1', 'train_2973_2', 'train_2973_3', 'train_2973_4', 'train_2973_5', 'train_2973_6', 'train_2973_7', 'train_2973_8', 'train_3036', 'train_3036_0', 'train_3036_1', 'train_3036_2', 'train_3036_3', 'train_3036_4', 'train_3036_5', 'train_3036_6', 'train_3036_7', 'train_3036_8', 'train_3070', 'train_3070_0', 'train_3070_1', 'train_3070_2', 'train_3070_3', 'train_3070_4', 'train_3070_5', 'train_3070_6', 'train_3070_7', 'train_3070_8', 'train_3093', 'train_3093_0', 'train_3093_1', 'train_3093_2', 'train_3093_3', 'train_3093_4', 'train_3093_5', 'train_3093_6', 'train_3093_7', 'train_3093_8', 'train_3212', 'train_3212_0', 'train_3212_1', 'train_3212_2', 'train_3212_3', 'train_3212_4', 'train_3212_5', 'train_3212_6', 'train_3212_7', 'train_3212_8', 'train_3214', 'train_3214_0', 'train_3214_1', 'train_3214_2', 'train_3214_3', 'train_3214_4', 'train_3214_5', 'train_3214_6', 'train_3214_7', 'train_3214_8', 'train_3251', 'train_3251_0', 'train_3251_1', 'train_3251_2', 'train_3251_3', 'train_3251_4', 'train_3251_5', 'train_3251_6', 'train_3251_7', 'train_3251_8', 'train_3266', 'train_3266_0', 'train_3266_1', 'train_3266_2', 'train_3266_3', 'train_3266_4', 'train_3266_5', 'train_3266_6', 'train_3266_7', 'train_3266_8', 'train_3269', 'train_3269_0', 'train_3269_1', 'train_3269_2', 'train_3269_3', 'train_3269_4', 'train_3269_5', 'train_3269_6', 'train_3269_7', 'train_3269_8', 'train_3340', 'train_3340_0', 'train_3340_1', 'train_3340_2', 'train_3340_3', 'train_3340_4', 'train_3340_5', 'train_3340_6', 'train_3340_7', 'train_3340_8', 'train_3341', 'train_3341_0', 'train_3341_1', 'train_3341_2', 'train_3341_3', 'train_3341_4', 'train_3341_5', 'train_3341_6', 'train_3341_7', 'train_3341_8', 'train_3364', 'train_3364_0', 'train_3364_1', 'train_3364_2', 'train_3364_3', 'train_3364_4', 'train_3364_5', 'train_3364_6', 'train_3364_7', 'train_3364_8', 'train_3443', 'train_3443_0', 'train_3443_1', 'train_3443_2', 'train_3443_3', 'train_3443_4', 'train_3443_5', 'train_3443_6', 'train_3443_7', 'train_3443_8', 'train_3451', 'train_3451_0', 'train_3451_1', 'train_3451_2', 'train_3451_3', 'train_3451_4', 'train_3451_5', 'train_3451_6', 'train_3451_7', 'train_3451_8', 'train_3454', 'train_3454_0', 'train_3454_1', 'train_3454_2', 'train_3454_3', 'train_3454_4', 'train_3454_5', 'train_3454_6', 'train_3454_7', 'train_3454_8', 'train_3542', 'train_3542_0', 'train_3542_1', 'train_3542_2', 'train_3542_3', 'train_3542_4', 'train_3542_5', 'train_3542_6', 'train_3542_7', 'train_3542_8', 'train_3547', 'train_3547_0', 'train_3547_1', 'train_3547_2', 'train_3547_3', 'train_3547_4', 'train_3547_5', 'train_3547_6', 'train_3547_7', 'train_3547_8', 'train_3548', 'train_3548_0', 'train_3548_1', 'train_3548_2', 'train_3548_3', 'train_3548_4', 'train_3548_5', 'train_3548_6', 'train_3548_7', 'train_3548_8', 'train_3560', 'train_3560_0', 'train_3560_1', 'train_3560_2', 'train_3560_3', 'train_3560_4', 'train_3560_5', 'train_3560_6', 'train_3560_7', 'train_3560_8', 'train_3574', 'train_3574_0', 'train_3574_1', 'train_3574_2', 'train_3574_3', 'train_3574_4', 'train_3574_5', 'train_3574_6', 'train_3574_7', 'train_3574_8', 'train_3620', 'train_3620_0', 'train_3620_1', 'train_3620_2', 'train_3620_3', 'train_3620_4', 'train_3620_5', 'train_3620_6', 'train_3620_7', 'train_3620_8', 'train_3627', 'train_3627_0', 'train_3627_1', 'train_3627_2', 'train_3627_3', 'train_3627_4', 'train_3627_5', 'train_3627_6', 'train_3627_7', 'train_3627_8', 'train_3636', 'train_3636_0', 'train_3636_1', 'train_3636_2', 'train_3636_3', 'train_3636_4', 'train_3636_5', 'train_3636_6', 'train_3636_7', 'train_3636_8', 'train_3655', 'train_3655_0', 'train_3655_1', 'train_3655_2', 'train_3655_3', 'train_3655_4', 'train_3655_5', 'train_3655_6', 'train_3655_7', 'train_3655_8', 'train_3718', 'train_3718_0', 'train_3718_1', 'train_3718_2', 'train_3718_3', 'train_3718_4', 'train_3718_5', 'train_3718_6', 'train_3718_7', 'train_3718_8', 'train_3725', 'train_3725_0', 'train_3725_1', 'train_3725_2', 'train_3725_3', 'train_3725_4', 'train_3725_5', 'train_3725_6', 'train_3725_7', 'train_3725_8', 'train_3816', 'train_3816_0', 'train_3816_1', 'train_3816_2', 'train_3816_3', 'train_3816_4', 'train_3816_5', 'train_3816_6', 'train_3816_7', 'train_3816_8', 'train_3825', 'train_3825_0', 'train_3825_1', 'train_3825_2', 'train_3825_3', 'train_3825_4', 'train_3825_5', 'train_3825_6', 'train_3825_7', 'train_3825_8', 'train_3827', 'train_3827_0', 'train_3827_1', 'train_3827_2', 'train_3827_3', 'train_3827_4', 'train_3827_5', 'train_3827_6', 'train_3827_7', 'train_3827_8', 'train_4505', 'train_4505_0', 'train_4505_1', 'train_4505_2', 'train_4505_3', 'train_4505_4', 'train_4505_5', 'train_4505_6', 'train_4505_7', 'train_4505_8', 'train_4540', 'train_4540_0', 'train_4540_1', 'train_4540_2', 'train_4540_3', 'train_4540_4', 'train_4540_5', 'train_4540_6', 'train_4540_7', 'train_4540_8', 'train_4671', 'train_4671_0', 'train_4671_1', 'train_4671_2', 'train_4671_3', 'train_4671_4', 'train_4671_5', 'train_4671_6', 'train_4671_7', 'train_4671_8', 'train_4757', 'train_4757_0', 'train_4757_1', 'train_4757_2', 'train_4757_3', 'train_4757_4', 'train_4757_5', 'train_4757_6', 'train_4757_7', 'train_4757_8', 'train_4767', 'train_4767_0', 'train_4767_1', 'train_4767_2', 'train_4767_3', 'train_4767_4', 'train_4767_5', 'train_4767_6', 'train_4767_7', 'train_4767_8', 'train_4802', 'train_4802_0', 'train_4802_1', 'train_4802_2', 'train_4802_3', 'train_4802_4', 'train_4802_5', 'train_4802_6', 'train_4802_7', 'train_4802_8', 'train_4826', 'train_4826_0', 'train_4826_1', 'train_4826_2', 'train_4826_3', 'train_4826_4', 'train_4826_5', 'train_4826_6', 'train_4826_7', 'train_4826_8', 'train_4995', 'train_4995_0', 'train_4995_1', 'train_4995_2', 'train_4995_3', 'train_4995_4', 'train_4995_5', 'train_4995_6', 'train_4995_7', 'train_4995_8', 'train_5058', 'train_5058_0', 'train_5058_1', 'train_5058_2', 'train_5058_3', 'train_5058_4', 'train_5058_5', 'train_5058_6', 'train_5058_7', 'train_5058_8', 'train_5076', 'train_5076_0', 'train_5076_1', 'train_5076_2', 'train_5076_3', 'train_5076_4', 'train_5076_5', 'train_5076_6', 'train_5076_7', 'train_5076_8', 'train_5155', 'train_5155_0', 'train_5155_1', 'train_5155_2', 'train_5155_3', 'train_5155_4', 'train_5155_5', 'train_5155_6', 'train_5155_7', 'train_5155_8', 'train_5162', 'train_5162_0', 'train_5162_1', 'train_5162_2', 'train_5162_3', 'train_5162_4', 'train_5162_5', 'train_5162_6', 'train_5162_7', 'train_5162_8', 'train_5339', 'train_5339_0', 'train_5339_1', 'train_5339_2', 'train_5339_3', 'train_5339_4', 'train_5339_5', 'train_5339_6', 'train_5339_7', 'train_5339_8', 'train_5373', 'train_5373_0', 'train_5373_1', 'train_5373_2', 'train_5373_3', 'train_5373_4', 'train_5373_5', 'train_5373_6', 'train_5373_7', 'train_5373_8', 'train_5550', 'train_5550_0', 'train_5550_1', 'train_5550_2', 'train_5550_3', 'train_5550_4', 'train_5550_5', 'train_5550_6', 'train_5550_7', 'train_5550_8', 'train_5551', 'train_5551_0', 'train_5551_1', 'train_5551_2', 'train_5551_3', 'train_5551_4', 'train_5551_5', 'train_5551_6', 'train_5551_7', 'train_5551_8', 'train_5621', 'train_5621_0', 'train_5621_1', 'train_5621_2', 'train_5621_3', 'train_5621_4', 'train_5621_5', 'train_5621_6', 'train_5621_7', 'train_5621_8', 'train_5726', 'train_5726_0', 'train_5726_1', 'train_5726_2', 'train_5726_3', 'train_5726_4', 'train_5726_5', 'train_5726_6', 'train_5726_7', 'train_5726_8', 'train_5783', 'train_5783_0', 'train_5783_1', 'train_5783_2', 'train_5783_3', 'train_5783_4', 'train_5783_5', 'train_5783_6', 'train_5783_7', 'train_5783_8', 'train_5793', 'train_5793_0', 'train_5793_1', 'train_5793_2', 'train_5793_3', 'train_5793_4', 'train_5793_5', 'train_5793_6', 'train_5793_7', 'train_5793_8', 'train_5801', 'train_5801_0', 'train_5801_1', 'train_5801_2', 'train_5801_3', 'train_5801_4', 'train_5801_5', 'train_5801_6', 'train_5801_7', 'train_5801_8', 'train_5849', 'train_5849_0', 'train_5849_1', 'train_5849_2', 'train_5849_3', 'train_5849_4', 'train_5849_5', 'train_5849_6', 'train_5849_7', 'train_5849_8', 'train_5864', 'train_5864_0', 'train_5864_1', 'train_5864_2', 'train_5864_3', 'train_5864_4', 'train_5864_5', 'train_5864_6', 'train_5864_7', 'train_5864_8', 'train_5897', 'train_5897_0', 'train_5897_1', 'train_5897_2', 'train_5897_3', 'train_5897_4', 'train_5897_5', 'train_5897_6', 'train_5897_7', 'train_5897_8', 'train_5906', 'train_5906_0', 'train_5906_1', 'train_5906_2', 'train_5906_3', 'train_5906_4', 'train_5906_5', 'train_5906_6', 'train_5906_7', 'train_5906_8'])
        b[1]['val']=np.array(['train_6077', 'train_6172', 'train_6285', 'train_6378', 'train_6494', 'train_6890', 'train_6933', 'train_6937', 'train_6995', 'train_7050', 'train_7114', 'train_7149', 'train_7234', 'train_7327', 'train_7418', 'train_7498', 'train_7515', 'train_7546', 'train_7572', 'train_7625', 'train_7662', 'train_7663', 'train_7664', 'train_7756', 'train_7778', 'train_7997', 'train_8297'])

        b[2]['train'] = np.array(['train_2855', 'train_2855_0', 'train_2855_1', 'train_2855_2', 'train_2855_3', 'train_2855_4', 'train_2855_5', 'train_2855_6', 'train_2855_7', 'train_2855_8', 'train_2973', 'train_2973_0', 'train_2973_1', 'train_2973_2', 'train_2973_3', 'train_2973_4', 'train_2973_5', 'train_2973_6', 'train_2973_7', 'train_2973_8', 'train_3036', 'train_3036_0', 'train_3036_1', 'train_3036_2', 'train_3036_3', 'train_3036_4', 'train_3036_5', 'train_3036_6', 'train_3036_7', 'train_3036_8', 'train_3070', 'train_3070_0', 'train_3070_1', 'train_3070_2', 'train_3070_3', 'train_3070_4', 'train_3070_5', 'train_3070_6', 'train_3070_7', 'train_3070_8', 'train_3093', 'train_3093_0', 'train_3093_1', 'train_3093_2', 'train_3093_3', 'train_3093_4', 'train_3093_5', 'train_3093_6', 'train_3093_7', 'train_3093_8', 'train_3212', 'train_3212_0', 'train_3212_1', 'train_3212_2', 'train_3212_3', 'train_3212_4', 'train_3212_5', 'train_3212_6', 'train_3212_7', 'train_3212_8', 'train_3214', 'train_3214_0', 'train_3214_1', 'train_3214_2', 'train_3214_3', 'train_3214_4', 'train_3214_5', 'train_3214_6', 'train_3214_7', 'train_3214_8', 'train_3251', 'train_3251_0', 'train_3251_1', 'train_3251_2', 'train_3251_3', 'train_3251_4', 'train_3251_5', 'train_3251_6', 'train_3251_7', 'train_3251_8', 'train_3266', 'train_3266_0', 'train_3266_1', 'train_3266_2', 'train_3266_3', 'train_3266_4', 'train_3266_5', 'train_3266_6', 'train_3266_7', 'train_3266_8', 'train_3269', 'train_3269_0', 'train_3269_1', 'train_3269_2', 'train_3269_3', 'train_3269_4', 'train_3269_5', 'train_3269_6', 'train_3269_7', 'train_3269_8', 'train_3340', 'train_3340_0', 'train_3340_1', 'train_3340_2', 'train_3340_3', 'train_3340_4', 'train_3340_5', 'train_3340_6', 'train_3340_7', 'train_3340_8', 'train_3341', 'train_3341_0', 'train_3341_1', 'train_3341_2', 'train_3341_3', 'train_3341_4', 'train_3341_5', 'train_3341_6', 'train_3341_7', 'train_3341_8', 'train_3364', 'train_3364_0', 'train_3364_1', 'train_3364_2', 'train_3364_3', 'train_3364_4', 'train_3364_5', 'train_3364_6', 'train_3364_7', 'train_3364_8', 'train_3443', 'train_3443_0', 'train_3443_1', 'train_3443_2', 'train_3443_3', 'train_3443_4', 'train_3443_5', 'train_3443_6', 'train_3443_7', 'train_3443_8', 'train_3451', 'train_3451_0', 'train_3451_1', 'train_3451_2', 'train_3451_3', 'train_3451_4', 'train_3451_5', 'train_3451_6', 'train_3451_7', 'train_3451_8', 'train_3454', 'train_3454_0', 'train_3454_1', 'train_3454_2', 'train_3454_3', 'train_3454_4', 'train_3454_5', 'train_3454_6', 'train_3454_7', 'train_3454_8', 'train_3542', 'train_3542_0', 'train_3542_1', 'train_3542_2', 'train_3542_3', 'train_3542_4', 'train_3542_5', 'train_3542_6', 'train_3542_7', 'train_3542_8', 'train_3547', 'train_3547_0', 'train_3547_1', 'train_3547_2', 'train_3547_3', 'train_3547_4', 'train_3547_5', 'train_3547_6', 'train_3547_7', 'train_3547_8', 'train_3548', 'train_3548_0', 'train_3548_1', 'train_3548_2', 'train_3548_3', 'train_3548_4', 'train_3548_5', 'train_3548_6', 'train_3548_7', 'train_3548_8', 'train_3560', 'train_3560_0', 'train_3560_1', 'train_3560_2', 'train_3560_3', 'train_3560_4', 'train_3560_5', 'train_3560_6', 'train_3560_7', 'train_3560_8', 'train_3964', 'train_3964_0', 'train_3964_1', 'train_3964_2', 'train_3964_3', 'train_3964_4', 'train_3964_5', 'train_3964_6', 'train_3964_7', 'train_3964_8', 'train_3970', 'train_3970_0', 'train_3970_1', 'train_3970_2', 'train_3970_3', 'train_3970_4', 'train_3970_5', 'train_3970_6', 'train_3970_7', 'train_3970_8', 'train_4105', 'train_4105_0', 'train_4105_1', 'train_4105_2', 'train_4105_3', 'train_4105_4', 'train_4105_5', 'train_4105_6', 'train_4105_7', 'train_4105_8', 'train_4148', 'train_4148_0', 'train_4148_1', 'train_4148_2', 'train_4148_3', 'train_4148_4', 'train_4148_5', 'train_4148_6', 'train_4148_7', 'train_4148_8', 'train_4265', 'train_4265_0', 'train_4265_1', 'train_4265_2', 'train_4265_3', 'train_4265_4', 'train_4265_5', 'train_4265_6', 'train_4265_7', 'train_4265_8', 'train_4363', 'train_4363_0', 'train_4363_1', 'train_4363_2', 'train_4363_3', 'train_4363_4', 'train_4363_5', 'train_4363_6', 'train_4363_7', 'train_4363_8', 'train_4383', 'train_4383_0', 'train_4383_1', 'train_4383_2', 'train_4383_3', 'train_4383_4', 'train_4383_5', 'train_4383_6', 'train_4383_7', 'train_4383_8', 'train_4485', 'train_4485_0', 'train_4485_1', 'train_4485_2', 'train_4485_3', 'train_4485_4', 'train_4485_5', 'train_4485_6', 'train_4485_7', 'train_4485_8', 'train_4486', 'train_4486_0', 'train_4486_1', 'train_4486_2', 'train_4486_3', 'train_4486_4', 'train_4486_5', 'train_4486_6', 'train_4486_7', 'train_4486_8', 'train_4487', 'train_4487_0', 'train_4487_1', 'train_4487_2', 'train_4487_3', 'train_4487_4', 'train_4487_5', 'train_4487_6', 'train_4487_7', 'train_4487_8', 'train_4505', 'train_4505_0', 'train_4505_1', 'train_4505_2', 'train_4505_3', 'train_4505_4', 'train_4505_5', 'train_4505_6', 'train_4505_7', 'train_4505_8', 'train_4540', 'train_4540_0', 'train_4540_1', 'train_4540_2', 'train_4540_3', 'train_4540_4', 'train_4540_5', 'train_4540_6', 'train_4540_7', 'train_4540_8', 'train_4671', 'train_4671_0', 'train_4671_1', 'train_4671_2', 'train_4671_3', 'train_4671_4', 'train_4671_5', 'train_4671_6', 'train_4671_7', 'train_4671_8', 'train_4757', 'train_4757_0', 'train_4757_1', 'train_4757_2', 'train_4757_3', 'train_4757_4', 'train_4757_5', 'train_4757_6', 'train_4757_7', 'train_4757_8', 'train_4767', 'train_4767_0', 'train_4767_1', 'train_4767_2', 'train_4767_3', 'train_4767_4', 'train_4767_5', 'train_4767_6', 'train_4767_7', 'train_4767_8', 'train_4802', 'train_4802_0', 'train_4802_1', 'train_4802_2', 'train_4802_3', 'train_4802_4', 'train_4802_5', 'train_4802_6', 'train_4802_7', 'train_4802_8', 'train_4826', 'train_4826_0', 'train_4826_1', 'train_4826_2', 'train_4826_3', 'train_4826_4', 'train_4826_5', 'train_4826_6', 'train_4826_7', 'train_4826_8', 'train_4995', 'train_4995_0', 'train_4995_1', 'train_4995_2', 'train_4995_3', 'train_4995_4', 'train_4995_5', 'train_4995_6', 'train_4995_7', 'train_4995_8', 'train_5058', 'train_5058_0', 'train_5058_1', 'train_5058_2', 'train_5058_3', 'train_5058_4', 'train_5058_5', 'train_5058_6', 'train_5058_7', 'train_5058_8', 'train_5076', 'train_5076_0', 'train_5076_1', 'train_5076_2', 'train_5076_3', 'train_5076_4', 'train_5076_5', 'train_5076_6', 'train_5076_7', 'train_5076_8', 'train_5155', 'train_5155_0', 'train_5155_1', 'train_5155_2', 'train_5155_3', 'train_5155_4', 'train_5155_5', 'train_5155_6', 'train_5155_7', 'train_5155_8', 'train_5162', 'train_5162_0', 'train_5162_1', 'train_5162_2', 'train_5162_3', 'train_5162_4', 'train_5162_5', 'train_5162_6', 'train_5162_7', 'train_5162_8', 'train_5339', 'train_5339_0', 'train_5339_1', 'train_5339_2', 'train_5339_3', 'train_5339_4', 'train_5339_5', 'train_5339_6', 'train_5339_7', 'train_5339_8', 'train_5373', 'train_5373_0', 'train_5373_1', 'train_5373_2', 'train_5373_3', 'train_5373_4', 'train_5373_5', 'train_5373_6', 'train_5373_7', 'train_5373_8', 'train_5550', 'train_5550_0', 'train_5550_1', 'train_5550_2', 'train_5550_3', 'train_5550_4', 'train_5550_5', 'train_5550_6', 'train_5550_7', 'train_5550_8', 'train_5551', 'train_5551_0', 'train_5551_1', 'train_5551_2', 'train_5551_3', 'train_5551_4', 'train_5551_5', 'train_5551_6', 'train_5551_7', 'train_5551_8', 'train_5621', 'train_5621_0', 'train_5621_1', 'train_5621_2', 'train_5621_3', 'train_5621_4', 'train_5621_5', 'train_5621_6', 'train_5621_7', 'train_5621_8', 'train_5726', 'train_5726_0', 'train_5726_1', 'train_5726_2', 'train_5726_3', 'train_5726_4', 'train_5726_5', 'train_5726_6', 'train_5726_7', 'train_5726_8', 'train_5783', 'train_5783_0', 'train_5783_1', 'train_5783_2', 'train_5783_3', 'train_5783_4', 'train_5783_5', 'train_5783_6', 'train_5783_7', 'train_5783_8', 'train_5793', 'train_5793_0', 'train_5793_1', 'train_5793_2', 'train_5793_3', 'train_5793_4', 'train_5793_5', 'train_5793_6', 'train_5793_7', 'train_5793_8', 'train_5801', 'train_5801_0', 'train_5801_1', 'train_5801_2', 'train_5801_3', 'train_5801_4', 'train_5801_5', 'train_5801_6', 'train_5801_7', 'train_5801_8', 'train_5849', 'train_5849_0', 'train_5849_1', 'train_5849_2', 'train_5849_3', 'train_5849_4', 'train_5849_5', 'train_5849_6', 'train_5849_7', 'train_5849_8', 'train_5864', 'train_5864_0', 'train_5864_1', 'train_5864_2', 'train_5864_3', 'train_5864_4', 'train_5864_5', 'train_5864_6', 'train_5864_7', 'train_5864_8', 'train_5897', 'train_5897_0', 'train_5897_1', 'train_5897_2', 'train_5897_3', 'train_5897_4', 'train_5897_5', 'train_5897_6', 'train_5897_7', 'train_5897_8', 'train_5906', 'train_5906_0', 'train_5906_1', 'train_5906_2', 'train_5906_3', 'train_5906_4', 'train_5906_5', 'train_5906_6', 'train_5906_7', 'train_5906_8'])
        b[2]['val'] = np.array(
            ['train_6077', 'train_6172', 'train_6285', 'train_6378', 'train_6494', 'train_6890', 'train_6933',
             'train_6937', 'train_6995', 'train_7050', 'train_7114', 'train_7149', 'train_7234', 'train_7327',
             'train_7418', 'train_7498', 'train_7515', 'train_7546', 'train_7572', 'train_7625', 'train_7662',
             'train_7663', 'train_7664', 'train_7756', 'train_7778', 'train_7997', 'train_8297'])

        save_pickle(b,split_file)
        splits = load_pickle(split_file)
        print(splits)
        
    elif args.task=='1':
        input_file = './DATASET/nnFormer_preprocessed/Task001_ACDC/nnFormerPlansv2.1_plans_3D.pkl'
        output_file = './DATASET/nnFormer_preprocessed/Task001_ACDC/nnFormerPlansv2.1_ACDC_plans_3D.pkl'
        a = load_pickle(input_file)
        
        a['plans_per_stage'][0]['patch_size']=np.array([14,160,160])
        a['plans_per_stage'][0]['pool_op_kernel_sizes']=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
        a['plans_per_stage'][0]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        
        split_file=input_file.replace('nnFormerPlansv2.1_plans_3D','splits_final')
        b = load_pickle(split_file)
        b[0]['train']=np.array(['patient001_frame01', 'patient001_frame12', 'patient004_frame01',
       'patient004_frame15', 'patient005_frame01', 'patient005_frame13',
       'patient006_frame01', 'patient006_frame16', 'patient007_frame01',
       'patient007_frame07', 'patient010_frame01', 'patient010_frame13',
       'patient011_frame01', 'patient011_frame08', 'patient013_frame01',
       'patient013_frame14', 'patient015_frame01', 'patient015_frame10',
       'patient016_frame01', 'patient016_frame12', 'patient018_frame01',
       'patient018_frame10', 'patient019_frame01', 'patient019_frame11',
       'patient020_frame01', 'patient020_frame11', 'patient021_frame01',
       'patient021_frame13', 'patient022_frame01', 'patient022_frame11',
       'patient023_frame01', 'patient023_frame09', 'patient025_frame01',
       'patient025_frame09', 'patient026_frame01', 'patient026_frame12',
       'patient027_frame01', 'patient027_frame11', 'patient028_frame01',
       'patient028_frame09', 'patient029_frame01', 'patient029_frame12',
       'patient030_frame01', 'patient030_frame12', 'patient031_frame01',
       'patient031_frame10', 'patient032_frame01', 'patient032_frame12',
       'patient033_frame01', 'patient033_frame14', 'patient034_frame01',
       'patient034_frame16', 'patient035_frame01', 'patient035_frame11',
       'patient036_frame01', 'patient036_frame12', 'patient037_frame01',
       'patient037_frame12', 'patient038_frame01', 'patient038_frame11',
       'patient039_frame01', 'patient039_frame10', 'patient040_frame01',
       'patient040_frame13', 'patient041_frame01', 'patient041_frame11',
       'patient043_frame01', 'patient043_frame07', 'patient044_frame01',
       'patient044_frame11', 'patient045_frame01', 'patient045_frame13',
       'patient046_frame01', 'patient046_frame10', 'patient047_frame01',
       'patient047_frame09', 'patient050_frame01', 'patient050_frame12',
       'patient051_frame01', 'patient051_frame11', 'patient052_frame01',
       'patient052_frame09', 'patient054_frame01', 'patient054_frame12',
       'patient056_frame01', 'patient056_frame12', 'patient057_frame01',
       'patient057_frame09', 'patient058_frame01', 'patient058_frame14',
       'patient059_frame01', 'patient059_frame09', 'patient060_frame01',
       'patient060_frame14', 'patient061_frame01', 'patient061_frame10',
       'patient062_frame01', 'patient062_frame09', 'patient063_frame01',
       'patient063_frame16', 'patient065_frame01', 'patient065_frame14',
       'patient066_frame01', 'patient066_frame11', 'patient068_frame01',
       'patient068_frame12', 'patient069_frame01', 'patient069_frame12',
       'patient070_frame01', 'patient070_frame10', 'patient071_frame01',
       'patient071_frame09', 'patient072_frame01', 'patient072_frame11',
       'patient073_frame01', 'patient073_frame10', 'patient074_frame01',
       'patient074_frame12', 'patient075_frame01', 'patient075_frame06',
       'patient076_frame01', 'patient076_frame12', 'patient077_frame01',
       'patient077_frame09', 'patient078_frame01', 'patient078_frame09',
       'patient080_frame01', 'patient080_frame10', 'patient082_frame01',
       'patient082_frame07', 'patient083_frame01', 'patient083_frame08',
       'patient084_frame01', 'patient084_frame10', 'patient085_frame01',
       'patient085_frame09', 'patient086_frame01', 'patient086_frame08',
       'patient087_frame01', 'patient087_frame10'])
        
        
        b[0]['val']=np.array(['patient089_frame01', 'patient089_frame10', 'patient090_frame04',
       'patient090_frame11', 'patient091_frame01', 'patient091_frame09',
       'patient093_frame01', 'patient093_frame14', 'patient094_frame01',
       'patient094_frame07', 'patient096_frame01', 'patient096_frame08',
       'patient097_frame01', 'patient097_frame11', 'patient098_frame01',
       'patient098_frame09', 'patient099_frame01', 'patient099_frame09',
       'patient100_frame01', 'patient100_frame13'])
        save_pickle(b,split_file)
        save_pickle(a, output_file)
    print(output_file)
