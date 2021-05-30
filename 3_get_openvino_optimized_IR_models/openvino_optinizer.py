# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""This script get openvino optimized model"""
# =============================================================================
# Import Required Libraries
# =============================================================================
import os
import platform
is_win = 'windows' in platform.platform().lower()

# OpenVINO 2019
if is_win:
    mo_tf_path = r'"C:\Program Files (x86)\IntelSWTools\openvino_2020.3.341\deployment_tools\model_optimizer\mo_tf.py"'
else:
    # mo_tf.py path in Linux
    mo_tf_path = '/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py'


pb_file = r'"./../Models/2_tf_frozen_PB_model/frozen_model.pb"'
output_dir = r'"./../Models/3_openvino_IR_model"'
img_height = 28
input_shape = [1,img_height,img_height,1]
input_shape_str = str(input_shape).replace(' ','')
print (input_shape_str)

command = 'python {} --input_model {} --output_dir {} --input_shape {} --data_type FP32'.format(mo_tf_path,pb_file,output_dir,input_shape_str)

os.system(command)
