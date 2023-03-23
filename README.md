# [NTIRE 2023 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2023/) @ [CVPR 2023](https://cvpr2023.thecvf.com/)

## Environment
we used PyTorch version 1.13.1 cuda version 11.7

## How to test the our model?

1. `git clone https://github.com/UESTCIPLAB/ARFT_for_NTIRE2023_Image_Super_Resolution.git`
2.   Pre-trained model : [Google Spreadsheet](https://drive.google.com/file/d/1fUDMXzHOornW4nDIxwAZbz3MynVBIKZP/view?usp=share_link)
3.   result : [Google Spreadsheet](https://drive.google.com/file/d/1bW3FZLISpv10XxxAi1iAvx2ffWP8sPQG/view?usp=share_link)
4. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 5
    ```

## How to add your model to this baseline?
1. Register your team in the Google Spreadsheet and get our model file.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
   - Note:  Please provide a download link for the pretrained model, if the file size exceeds **100 MB**. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].txt`: e.g. [team00_cat.txt](https://github.com/zhengchen1999/NTIRE2023_ImageSR_x4/blob/main/model_zoo/team00_cat.txt)
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   
## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RFDN import RFDN
    model = RFDN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```
## Acknowledgement
This code is bulit on [ART](https://github.com/gladzhang/ART) codebase. We thank the authors for sharing the codes


## License
This code repository is release under [MIT License](LICENSE). 
