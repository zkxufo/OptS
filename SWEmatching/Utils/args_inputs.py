import argparse

def getArgs():
    parser = argparse.ArgumentParser(description="All Parameters needed")
    #  Deep Learning Parameters
    parser.add_argument('--Model', type=str, default="Alexnet", help='Model Name')
    parser.add_argument('--split', type=str, default="val", help='val or train')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size in each interation')
    parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda:0')
    parser.add_argument('--root', type=str, default="/home/h2amer/AhmedH.Salamah/ilsvrc2012", 
                            help='root to ImageNet Driectory')
    
    # ------------------------------------------------------------------------------------------------------------
    # Our Experimental Settings 

    parser.add_argument('--resize_compress', type=bool, default=False, help='For Resize --> Compress set True')
    parser.add_argument('--compress_resize', type=bool, default=False, help='For Compress --> Resize set True')
    parser.add_argument('--colorspace', type=int, default=0, help='ColorSpace 0:YUV')

    parser.add_argument('--machine_dist', type=bool, default=False, help='Include the Machine distortion')
    parser.add_argument('--output_txt', type=str, default="", help='output txt file')

    # Sensitivity Settings
    parser.add_argument('--beta', type=float, default=0, help='Beta to tune the sensitivity between Human and Machine \" 1+ Beta s_i\"')
    parser.add_argument('--SenMap_dir', type=str, default="./SenMap/", help='Senstivity Directory')

    # ------------------------------------------------------------------------------------------------------------
    
    #  HDQ Parameters
    parser.add_argument('--J', type=int, default=4, help='Subsampling J')
    parser.add_argument('--a', type=int, default=4, help='Subsampling a')
    parser.add_argument('--b', type=int, default=4, help='Subsampling b')
    parser.add_argument('--QF_Y', type=int, default=100, help='QF of Y channel')
    parser.add_argument('--QF_C', type=int, default=100, help='QF of C channel')
    # ------------------------------------------------------------------------------------------------------------
    #  OptD paramaters
    parser.add_argument('--OptD_enable', type=bool, default=False, help='OptD initialization for Quantization Table')
    parser.add_argument('--OptS_enable', type=bool, default=False, help='OptD initialization for Quantization Table')
    parser.add_argument('--JPEG_enable', type=bool, default=False, help='OptD initialization for Quantization Table')
    
    parser.add_argument('--Qmax_Y', type=int, default=46, help='Maximum Quantization Step Y Channel')
    parser.add_argument('--Qmax_C', type=int, default=46, help='Maximum Quantization Step C Channel')
    parser.add_argument('--d_waterlevel_Y', type=float, default=-1, help='Waterfilling level on Y channel')
    parser.add_argument('--d_waterlevel_C', type=float, default=-1, help='Waterfilling level on C channel')
    
    parser.add_argument('--DT_Y', type=float, default=1, help='Target Distortion on Y channel.')
    parser.add_argument('--DT_C', type=float, default=1, help='Target Distortion on C channel')

    args = parser.parse_args()
    return args