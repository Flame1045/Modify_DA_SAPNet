# Modify_DA_SAPNet
This is a modift SAPNet for Domain Adaption research
Modify by Eric Chiu

# Debug config
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": ["--config-file", "configs/sim10k2cityscapes/sapnetV2_R_50_C4.yaml", "--num-gpus", "1", "--setting-token", 
            "sim10k2city-sapnetV2-test", "INPUT.MIN_SIZE_TRAIN", "(500,)", 
            "MODEL.DA_HEAD.WINDOW_STRIDES","[2, 2, 2, 2, 2, 2, 2, 2, 2]",
            "MODEL.DA_HEAD.WINDOW_SIZES","[3, 6, 9, 12, 15, 18, 21, 24, 27]",
            "MODEL.DA_HEAD.LOSS_WEIGHT", "0.1", "MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT", "0.8",
            "MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT", "-0.3", "MODEL.WEIGHTS", "pretrained/sim10k-baseline/model_0023999.pth"]
        }
    ]
}


# Acknowledgement
* [SAPNetDA](https://isrc.iscas.ac.cn/gitlab/research/domain-adaption)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [Grad-CAM.pytorch](https://github.com/yizt/Grad-CAM.pytorch)  
* [Attentional Feature Fusion](https://github.com/YimianDai/open-aff)
* [MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation](https://github.com/lhoyer/MIC)
