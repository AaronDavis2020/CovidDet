from solution.args import args
import torch
from solution.dataGen import TestDataset
from tqdm import tqdm
import torch.utils.data as data
from solution.transform import get_transforms
from solution.buildNet import make_model
import pandas as pd


def slicePredict(use_cuda, dcm_list, dataset):
    # data
    transformations = get_transforms(input_size=args.image_size,test_size=args.image_size)
    test_set = TestDataset(dcm_list=dcm_list, dataset=dataset, transform= transformations['test']) # 测试图像名称列表
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
    # load model
    model = make_model(args)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)) # load model

    if use_cuda:
        model.cuda()

    # evaluate
    y_pred = []
    img_paths = []
    with torch.no_grad():
        model.eval() # swich the model into valid mode
        for (inputs, paths) in tqdm(test_loader):
            img_paths.extend(list(paths))
            if use_cuda:
                inputs = inputs.cuda()
            inputs = torch.autograd.Variable(inputs)
            # compute output
            outputs = model(inputs)  # (16,2)
            # probability = torch.nn.functional.softmax(outputs,dim=1)[:,1].tolist()
            # probability = [1 if prob >= 0.5 else 0 for prob in probability]
            # return the index of the max
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            try:
                y_pred.extend(probability)
            except TypeError:
                pass
        print("y_pred=",y_pred)

        res_dict = {
            'img_path':img_paths,
            'predict':y_pred,

        }
        df = pd.DataFrame(res_dict)
        # df.to_csv(args.result_csv,index=False)
        # print(f"write to {args.result_csv} succeed ")
        return df
