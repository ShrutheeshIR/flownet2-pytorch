import torch
import numpy as np
import argparse
from models import FlowNet2
from utils.frame_utils import read_gen 
import cvbase as cvb
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("checkpoints/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py

    
    val=0
    foldername = "datasets/dancelogue/frames/"
    filname = foldername + "output_"
    for idx in range(1250, 1750):

        img1name = filname + str(idx) + '.png'
        img2name = filname + str(idx+1) + '.png'
        pim1 = cvb.read_img(img1name)
        pim2 = cvb.read_img(img1name)
        pim1 = cvb.resize(pim1, (1920,1024), return_scale=False)
        pim2 = cvb.resize(pim2, (1920,1024), return_scale=False)
        images = [pim1, pim2]
        images = np.array(images).transpose(3, 0, 1, 2)
        print(images.shape)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        #print('HALLO')
        # process the image pair to obtian the flow
        t1 = time.time()
        result = net(im).squeeze()
        print(time.time()-t1)
        #val = val+time.time()
        

   
    # def writeFlow(name, flow):
    #     f = open(name, 'wb')
    #     f.write('PIEH'.encode('utf-8'))
    #     np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    #     flow = flow.astype(np.float32)
    #     print("flow's shape",flow.shape)
    #     flow.tofile(f)
    #     f.flush()
    #     f.close()
        data = result.data.cpu().numpy().transpose(1, 2, 0)
    # writeFlow("img0.flo", data)
    # flowf = cvb.read_flow('img0.flo')
        img = cvb.flow2rgb(data)
        opfilname = 'opp'+str(idx) + '.jpg'
        cvb.write_img(img*255., 'opimagesfolder/' + opfilname)
        

    print(time.time()-t1)
    