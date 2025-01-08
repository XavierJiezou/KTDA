import torch
import os
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str)
    parse.add_argument("--save_path", type=str)
    args = parse.parse_args()
    return args.weight_path,args.save_path,


def main():
    weight_path, save_path = get_args()
    weight = torch.load(weight_path, map_location="cpu")
    state_dict = weight["state_dict"]
    head_state_dict = {}
    auxiliary_head_dict = {}
    backbone_dict = {}
    neck_dict = {}
    student_adapter_dict = {}
    for k, v in state_dict.items():
        if "decode_head" in k:
            head_state_dict[k] = v
        elif "auxiliary_head" in k:
            auxiliary_head_dict[k] = v
        elif "backbone" in k:
            backbone_dict[k] = v
        elif "neck" in k:
            neck_dict[k] = v
        elif "student_adapter" in k:
            student_adapter_dict[k] = v
        else:
            raise ValueError(f"unexpected keys:{k}")
    torch.save(head_state_dict, os.path.join(save_path,"head.pth"))
    torch.save(auxiliary_head_dict, os.path.join(save_path,"auxiliary_head.pth"))
    torch.save(backbone_dict, os.path.join(save_path,"backbone.pth"))
    torch.save(neck_dict, os.path.join(save_path,"neck.pth"))
    torch.save(student_adapter_dict, os.path.join(save_path,"student_adapter.pth"))


if __name__ == "__main__":
    main()
